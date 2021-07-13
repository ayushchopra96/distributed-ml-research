wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data
rm tiny-imagenet-200.zip 
mkdir runs
mkdir stats
mkdir stats/cifar10
mkdir stats/tiny_imagenet
cd /tmp
git clone https://github.com/ojus1/TripletTorch
cd TripletTorch
python3 setup.py install