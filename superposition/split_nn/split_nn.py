from einops import rearrange
from einops import repeat
from fvcore.nn import FlopCountAnalysis
from .client import Clients
from .server import Server, PartitionedServer
import torch
from torch import nn


class SplitNN(nn.Module):
    def __init__(
        self,
        clients,
        server,
        clients_opt,
        server_opt,
        scheduler_list_alice,
        scheduler_bob,
        triplet_loss,
        clf_loss,
        interrupted=False,
        avg=False,
    ):
        super().__init__()
        self.clients = clients
        self.client_opts = clients_opt
        self.num_clients = len(self.clients.client_models)

        self.bob = server
        self.is_partitioned = False
        if isinstance(self.bob, PartitionedServer):
            self.is_partitioned = True
        self.opt_bob = server_opt

        self.to_server = None
        self.interrupted = interrupted
        self.scheduler_list_alice = scheduler_list_alice
        self.scheduler_bob = scheduler_bob
        self.avg = avg

        self.triplet = triplet_loss
        self.clf_loss = clf_loss

    def forward(self, inputs, labels, inputs_cl, labels_cl, selected_mask, flops, use_contrastive):
        b = inputs.shape[1]

        to_server, stump_out = self.clients(inputs)
        if use_contrastive:
            emb, _ = self.clients(inputs_cl)

        # flops += (
        #     FlopCountAnalysis(self.clients.client_models,
        #                       inputs=(inputs,))
        #     .unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        #     .total()
        # )

        final_out = self.bob(to_server, selected_mask)
        # flops += (
        #     FlopCountAnalysis(self.bob, inputs=(selected_mask, to_server))
        #     .unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        #     .total()
        # )

        # Losses
        stump_clf_loss = torch.zeros(labels.shape).cuda()
        triplet_loss = torch.zeros(labels.shape).cuda()
        final_clf_loss = torch.zeros(labels.shape).cuda()
        for i in range(self.num_clients):
            stump_clf_loss[i] = self.clf_loss(stump_out[i], labels[i])
            if use_contrastive:
                # print(labels_cl[i].shape, rearrange(
                #     emb[i], 'bc ch h w -> bc (ch h w)').shape)
                triplet_loss[i] = self.triplet(
                    labels_cl[i],
                    rearrange(
                        emb[i], 'bc ch h w -> bc (ch h w)')
                )[0].mean(0).reshape(b, -1).mean(-1)

            if not selected_mask[i]:
                final_clf_loss[i] = self.clf_loss(final_out[i], labels[i])

        return triplet_loss, stump_clf_loss, final_clf_loss, flops

    def infer(self, inputs):
        to_server, stump_out = self.clients(inputs)
        final_out = self.bob(to_server, torch.zeros(
            (inputs.shape[0], )).bool().cuda())
        return final_out, stump_out

    def get_grad(self):
        grad_to_client = self.bob.server_backward()
        return grad_to_client

    def backward(self, selected_mask) -> None:
        grad_to_client = self.bob.server_backward(selected_mask)
        self.clients.client_backward(grad_to_client, selected_mask)

    def zero_grads(self) -> None:
        self.opt_bob.zero_grad()
        for i in range(self.num_clients):
            self.client_opts[i].zero_grad()

    def train(self) -> None:
        self.bob.train()
        self.clients.train()

    def eval(self) -> None:
        self.bob.eval()
        self.clients.eval()

    def step(self) -> None:
        self.opt_bob.step()
        for i in range(self.num_clients):
            self.client_opts[i].step()

    def scheduler_step(self, acc_bob, out=True, step_bob=False):
        #         self.scheduler_list_alice[i].step(acc_bob)
        #         if out and step_bob:
        #             self.scheduler_bob.step(acc_bob)
        for i in range(self.num_clients):
            self.scheduler_list_alice[i].step()
        if out and step_bob:
            self.scheduler_bob.step()

    @torch.no_grad()
    def evaluator(self, test_loader):
        self.eval()
        if not self.is_partitioned:
            correct_m1, correct_m2 = 0.0, 0.0
            total = 0.0
            for images, labels in test_loader:
                images = repeat(
                    images.unsqueeze(0), "(clients) b c h w -> (repeat clients) b c h w", repeat=self.num_clients).cuda()
                labels = labels.cuda()

                output_final, output_stump = self.infer(images)
                # for i in range(self.num_clients):
                predicted_m1 = torch.argmax(output_final[0].data, 1)
                predicted_m2 = torch.argmax(output_stump[0].data, 1)

                correct_m1 += (predicted_m1 == labels).sum()
                correct_m2 += (predicted_m2 == labels).sum()

                total += labels.size(0)

            # accuracy of bob with ith alice
            accuracy_m1 = float(correct_m1) / total
            accuracy_m2 = float(correct_m2) / total  # accuracy of ith alice

            # print('Accuracy Model 1: %f %%' % (100 * accuracy_m1))
            # print('Accuracy Model 2: %f %%' % (100 * accuracy_m2))

            return accuracy_m1, accuracy_m2
        else:
            corrects_m1 = torch.zeros((self.num_clients,)).cuda()
            corrects_m2 = torch.zeros((self.num_clients,)).cuda()
            totals = torch.zeros((self.num_clients,)).cuda()
            for images, labels, mask in test_loader:
                images = images.cuda().transpose(1, 0)
                labels = labels.cuda().transpose(1, 0)
                mask = mask.cuda()
                output_final, output_stump = self.infer(images)
                for i in range(self.num_clients):
                    predicted_m1 = torch.argmax(output_final[i], 1)
                    predicted_m2 = torch.argmax(output_stump[i], 1)

                    corrects_m1[i] += torch.masked_select(
                        (predicted_m1 == labels), mask[:, i]).sum()
                    corrects_m2[i] += torch.masked_select(
                        (predicted_m2 == labels), mask[:, i]).sum()

                    totals[i] += torch.masked_select(labels,
                                                     mask[:, i]).size(0)

            # accuracy of bob with ith alice
            accuracy_m1 = (corrects_m1 / totals).mean()
            # accuracy of ith alice
            accuracy_m2 = (corrects_m2 / totals).mean()

            # print('Accuracy Model 1: %f %%' % (100 * accuracy_m1))
            # print('Accuracy Model 2: %f %%' % (100 * accuracy_m2))

            return accuracy_m1.item(), accuracy_m2.item()
