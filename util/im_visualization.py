import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def remove_white_edge(fig,height,width):

    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    

    
def image_show(out,china_path):
    '''
    out: k*182*232, wrong rate, k means PM25,O3,No2,So2
    china_path: china map
    '''

    size = (232*4,182*4)

    china = np.array(Image.open(china_path).resize(size))


    for i in range(out.shape[0]):
        fig, ax = plt.subplots()
        #out is wrong rate, 由于大部分的误差处在-100%~100%之间，所以设置了简单的clip，方便展示
        ax.imshow(out[i][::-1],vmin=-1, vmax=1, cmap='bwr',aspect='equal')
        plt.axis('off')
        height, width = out[i].shape
        remove_white_edge(fig,height,width)

        #dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
        plt.savefig('result.png', dpi=300)


        wrong_rate = np.array(Image.open('result.png').resize(size,Image.NEAREST))

        plt.figure(figsize=(18*0.7,23*0.7))

        cur = wrong_rate[:,:,:3]+china

        cur[0:4] = 254

        plt.imshow(cur)
        plt.show()
        
    