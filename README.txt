Accepted by IEEE TKDE

Install Keras v2.0, scikit-learn
sudo pip install keras scikit-learn

# settings in main.py

TEST = Ture
# when TEST = Ture, the code just test the trained SDMVC model

train_ae = False
# when train_ae = Ture, the code will pre-train the autoencoders first, and the fine-turn the model with SDMVC

data = 'BDGP'     
# the tested datasets contain:
# 'MNIST_USPS_COMIC'           (CAE)
# 'Sigle_Three_Fmnist_Test'    (CAE)
# 'BDGP'                       (FAE)
# 'Caltech101_20'              (FAE)

# run the code：
python main.py
