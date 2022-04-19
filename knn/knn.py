# %%
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# %%
img = Image.open('../imgs/flight_10.png')
data = str(img.tobitmap)

# %%
im = np.array(img)
im_shape = im.shape
# pil_image=Image.fromarray(im)
# pil_image.save('output.png')

# %%
im_arr = np.reshape(im, (im_shape[0]*im_shape[1], im_shape[2])) 
im_arr.shape

# %%
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
    # return "{}".format(rgb[0].tobytes())[4:6] + \
    #        "{}".format(rgb[1].tobytes())[4:6] + \
    #        "{}".format(rgb[2].tobytes())[4:6]
# %%
im_arrCol = np.array([rgb_to_hex(p) for p in im_arr])

# %%
np.random.seed(100)
train = np.random.choice(im_arr.shape[0], 2000, replace=False)


# %%
tsne = TSNE(n_components=2, 
            perplexity=30, 
            learning_rate='auto',
            init='random'
            )
tsne = tsne.fit_transform(im_arr[train])
# tsne.shape
# tsne.fit()


# %%
fig, ax = plt.subplots(1, figsize=(7,5))
ax.scatter(x=tsne[:,0], y=tsne[:,1], c=im_arrCol[train], s=3)
ax.set_aspect('equal')
plt.show()



# %%
