from PIL import Image
from numpy import asarray, unique, array_equal, all
an_image1 = Image.open("DD_full/SegmentationClass/fc5837dcf8_7CD52BE09EINSPIRE-000068.png")
an_image2 = Image.open("DD_full/SegmentationClassRaw/fc5837dcf8_7CD52BE09EINSPIRE-000068.png")
# an_image1 = Image.open("pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png")
# an_image2 = Image.open("pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw/2007_000032.png")

data = asarray(an_image1)
df = asarray(an_image2)
r,g,b = data[:,:,0], data[:,:,1], data[:,:,2]
# comparison = g == b
# equal_arrays = comparison.all()
equal_arrays = array_equal(g,b)
print(equal_arrays)
print('test')
print(r.shape)
# print('small image')
# print(data.shape)
# print(unique(data))
# print(data.dtype)
#
# print('large image')
# print(df.shape)
# print(unique(df))
# print(df.dtype)
