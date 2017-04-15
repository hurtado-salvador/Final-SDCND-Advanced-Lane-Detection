import glob
import matplotlib.pyplot as plt


imgPath1 = 'D:/aaSDCNDJ/Project4/imagenes/test*.jpg'
imgPath2 = 'D:/aaSDCNDJ/Project4/imagenes/warp*.PNG'
imgPath3 = 'D:/aaSDCNDJ/Project4/imagenes/binary*.PNG'
imgPath4 = 'D:/aaSDCNDJ/Project4/imagenes/lane*.PNG'
test = glob.glob(imgPath1)
warp = glob.glob(imgPath2)
binary = glob.glob(imgPath3)
lane = glob.glob(imgPath4)
'''
plt.figure(figsize=(9, 4))
plt.subplot(6, 1, 1)
img = plt.imread(test[0])
plt.imshow(img)
plt.title('Original image')

'''

def imagenes(gl1, gl2, gl3, gl4):

    for i1 in range(len(gl1)):
        plt.subplot(6,4,(i1*4)+1)
        img = plt.imread(gl1[i1])
        plt.imshow(img)
        #plt.title('Test ')
        plt.subplot(6, 4, (i1*4) + 2)
        img = plt.imread(gl2[i1])
        plt.imshow(img)
        #plt.title('Warp')
        plt.subplot(6, 4, (i1*4) + 3)
        img = plt.imread(gl3[i1])
        plt.imshow(img)
        #plt.title('Binary')
        plt.subplot(6, 4, (i1*4) + 4)
        img = plt.imread(gl4[i1])
        plt.imshow(img)
        #plt.title('Lane')


imagenes(test,warp,binary,lane)

plt.show()