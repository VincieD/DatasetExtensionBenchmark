# import numpy as np
#
# data= np.load('C:\\Users\\qxy9300\\Documents\\MA\\01_Dataset\\INRIA_Person_Dataset_Train\\DAGAN_ped_database.npy')
# # data = np.load('C:\\Users\\qxy9300\\Documents\\MA\\05_Program\\DAGAN-master\\datasets\\omniglot_data.npy')
#
# # a= data[0]
# # b=data[1]
# #
# # new_data = []
# #
# # for i, ar in enumerate(a):
# #     new_data.append(ar)
# # for i, ar in enumerate(b):
# #     new_data.append(ar)
#
#
# # new_data=np.empty([2,1526,128,128,3])
# print('hello')


# !/usr/bin/python

import os
for root, dirs, files in os.walk(os.path.join(os.getcwd(),'datasets','INRIA_Person_Dataset_Train_128')):
   # for name in files:
   #    print(os.path.join(root, name))
   ped = os.path.split(root)[-1]
   for name in dirs:
      print(os.path.join(root, name))
   # print('hello')