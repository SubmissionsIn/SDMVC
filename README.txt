Accepted by IEEE TKDE.

# citation
@ARTICLE{9839616,
  author={Xu, Jie and Ren, Yazhou and Tang, Huayi and Yang, Zhimeng and Pan, Lili and Yang, Yang and Pu, Xiaorong and Yu, Philip S. and He, Lifang},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Self-Supervised Discriminative Feature Learning for Deep Multi-View Clustering}, 
  year={2023},
  volume={35},
  number={7},
  pages={7470-7482},
  doi={10.1109/TKDE.2022.3193569}
}

Install Keras v2.0, scikit-learn
sudo pip install keras scikit-learn

# settings in main.py

TEST = Ture
# when TEST = Ture, the code just test the trained SDMVC model

train_ae = False
# when train_ae = Ture, the code will pre-train the autoencoders first, and the fine-turn the model with SDMVC

data = 'BDGP'     
# the tested datasets contain:
# 'MNIST_USPS'                 (CAE)
# 'Fashion_MV'                 (CAE)
# 'BDGP'                       (FAE)
# 'Caltech101_20'              (FAE)

# run the code：
python main.py

(SDMVC.7z is the old version uploaded in Feb. 2021)
