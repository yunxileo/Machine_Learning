"""
Programmer  :   EOF
File        :   E_Face.py
E-mail      :   jasonleaster@163.com
Date        :   2016.01.18
"""

from image import Image
from getCachedAdaBoost import getCachedAdaBoost

def scanImgFixedWin(image):
    assert isinstance(image, Image)

    ImgWidth  = image.Col
    ImgHeight = image.Row

    for x in range(ImgWidth - SEARCH_WIN_WIDTH):
        for y in range(ImgHeight - SEARCH_WIN_HEIGHT):
            searchWindow = (x, y, SEARCH_WIN_WIDTH, SEARCH_WIN_HEIGHT)


image = Image(Test_Img)

model = getCachedAdaBoost()


