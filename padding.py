import cv2,glob,os

desired_size = 250
directory='./cropped'
for file in os.listdir(directory):
    print(file)
    path2='/home/shitijkarsolia/Desktop/projects/Siamese/Working_Code/cropped/'+file
    print(path2)
    im_pth = path2

    im = cv2.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format
    # cv2.imshow("image", im)
    # cv2.waitKey(0)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    # cv2.imshow("image", new_im)
    val="/home/shitijkarsolia/Desktop/projects/Siamese/Working_Code/padding/"+file
    cv2.imwrite(val,new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()