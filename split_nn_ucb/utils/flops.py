from fvcore.nn import FlopCountAnalysis

flops_dict = {}
def compute_flops(model, args, alice_or_bob):
    if len(args) == 2:
        inputs, i = args
    else:
        inputs, = args

    memo_key = (alice_or_bob, str(list(inputs.shape)))
    if memo_key in flops_dict:
        return flops_dict[memo_key]

    f = FlopCountAnalysis(model, inputs=args).unsupported_ops_warnings(
        False).uncalled_modules_warnings(False).total()
    flops_dict[memo_key] = f
    return f
