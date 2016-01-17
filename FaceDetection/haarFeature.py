"""
Programmer  :   EOF
Date        :   2016.01.16
E-mail      :   jasonleaster@163.com
File        :   haarFeature.py

Decription:

    Types of Haar-like rectangle features
      
     --- ---      --- ---
    |   |   |    |   +   |
    | + | - |    |-------|
    |   |   |    |   -   |
     --- ---      -------
       A            B

     -- -- --     -------
    |  |  |  |   |___-___|       
    |- | +| -|   |___+___|
    |  |  |  |   |   -   |
     -- -- --     -------
        C            D

     --- ---
    | - | + |
    |___|___|
    | + | - |
    |___|___|
        E

    For each feature pattern, the start point(x, y) is at 
    the most left-up pixel in that window. The size of that
    window is @width * @height
"""
from vecProduct import VecProduct

class Feature:
    def __init__(self, win_Width, win_Height, img_Width, img_Height):

        self.featureName = "Haar Feature"

        self.win_Width  = win_Width
        self.win_Height = win_Height

        self.img_Width  = img_Width
        self.img_Height = img_Height

        self.tot_pixels = img_Width * img_Height

        self.featureTypes = ["I", "II", "III", "IV"]

        self.features   = self.evalFeatures()


    def vecRectSum(self, x, y, width, height):
        idxVector = [0 for i in range(self.tot_pixels)]
        if x == 0 and y == 0:
            idxVector[width * height + 2] = 1

        elif x == 0:
            idx1 = self.img_Height * (    width - 1) + height + y - 1
            idx2 = self.img_Height * (    width - 1) +          y - 1
            idxVector[idx1] = +1
            idxVector[idx2] = -1

        elif y == 0:
            idx1 = self.img_Height * (x + width - 1) + height - 1
            idx2 = self.img_Height * (x         - 1) + height - 1
            idxVector[idx1] = +1
            idxVector[idx2] = -1
        else:
            idx1 = self.img_Height * (x + width - 1) + height + y - 1
            idx2 = self.img_Height * (x + width - 1) +          y - 1
            idx3 = self.img_Height * (x         - 1) + height + y - 1
            idx4 = self.img_Height * (x         - 1) +          y - 1

            assert idx1 < self.tot_pixels and idx2 < self.tot_pixels 
            assert idx3 < self.tot_pixels and idx4 < self.tot_pixels 

            idxVector[idx1] = + 1
            idxVector[idx2] = - 1
            idxVector[idx3] = - 1
            idxVector[idx4] = + 1

        return idxVector


    def VecFeatureTypeI(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(x, y         , width, height)
        vec2 = self.vecRectSum(x, y + height, width, height)

        return VecProduct(vecImg, vec1) - \
               VecProduct(vecImg, vec2)

    def VecFeatureTypeII(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(x + width, y, width, height)
        vec2 = self.vecRectSum(x        , y, width, height)

        return VecProduct(vecImg, vec1) - \
               VecProduct(vecImg, vec2)

    def VecFeatureTypeIII(self,vecImg, x, y, width, height):
        vec1 = self.vecRectSum(x +   width, y, width, height)
        vec2 = self.vecRectSum(x          , y, width, height)
        vec3 = self.vecRectSum(x + 2*width, y, width, height)

        return VecProduct(vecImg, vec1) -\
               VecProduct(vecImg, vec2) -\
               VecProduct(vecImg, vec3)

    def VecFeatureTypeIV(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(x + width,          y, width, height)
        vec2 = self.vecRectSum(x        ,          y, width, height)
        vec3 = self.vecRectSum(x        , y + height, width, height)
        vec4 = self.vecRectSum(x + width, y + height, width, height)

        return VecProduct(vecImg, vec1) -\
               VecProduct(vecImg, vec2) +\
               VecProduct(vecImg, vec3) -\
               VecProduct(vecImg, vec4)
        
    def evalFeatures(self):
        win_Height = self.win_Height
        win_Width  = self.win_Width

        height_Limite = {"I"  : win_Height/2 - 1,
                         "II" : win_Height   - 1,
                         "III": win_Height   - 1,
                         "IV" : win_Height/2 - 1}

        width_Limit  = {"I"   : win_Width   - 1,
                        "II"  : win_Width/2 - 1,
                        "III" : win_Width/3 - 1,
                        "IV"  : win_Width/2 - 1}

        features = []
        for types in self.featureTypes:
            for h in range(1, height_Limite[types]):
                for w in range(1, width_Limit[types]):
                    if types == "I":
                        x_limit = win_Width  - w   
                        y_limit = win_Height - 2*h 
                    elif types == "II":
                        x_limit = win_Width  - 2*w 
                        y_limit = win_Height - h   
                    elif types == "III":
                        x_limit = win_Width  - 3*w 
                        y_limit = win_Height - h   
                    elif types == "IV":
                        x_limit = win_Width  - 2*w 
                        y_limit = win_Height - 2*h 

                    for x in range(1, x_limit):
                        for y in range(1, y_limit):
                            features.append( [types, x, y, w, h])

        return features


    def calFeatures(self, vecImg):
        # features[i] == [types, x, y, w, h]

        features = self.features

        for i in range(len(features)):
            (types, x, y, w, h) = [val for val in features[i]]

            if features[i][0] == "I":
                func = self.__VecFeatureTypeI
            elif features[i][0] == "II":
                func = self.__VecFeatureTypeII
            elif features[i][0] == "III":
                func = self.__VecFeatureTypeIII
            elif features[i][0] == "VI":
                func = self.__VecFeatureTypeIV
            else:
                raise ValueError("Undefined feature type!")

        raise ValueError("Unimplemented")
            #self.haarFeatureVal.append( func(vecImg, x, y, w, h))

    def makeFeaturePic(self, feature):
        (types, x, y, width, height) = [val for val in feature]

        assert x >= 0 and x < self.Row
        assert y >= 0 and y < self.Col
        assert width > 0 and height > 0

        image = numpy.array(self.img)

        if types == "I":
            for i in range(y, y + height * 2):
                for j in range(x, x + width):
                    if i < y + height:
                        image[i][j] = 255
                    else:
                        image[i][j] = 0

        elif types == "II":
            for i in range(y, y + height):
                for j in range(x, x + width * 2):
                    if j < x + width:
                        image[i][j] = 0
                    else:
                        image[i][j] = 255

        elif types == "III":
            for i in range(y, y + height):
                for j in range(x, x + width * 3):
                    if j >= (x + width) and j < (x + width * 2):
                        image[i][j] = 255
                    else:
                        image[i][j] = 0

        elif types == "IV":
            for i in range(y, y + height * 2):
                for j in range(x, x + width * 2):
                    if (j < x + width and i < y + height) or\
                       (j >= x + width and i >= y + height):
                        image[i][j] = 0
                    else:
                        image[i][j] = 255


        pylab.matshow(image, cmap = "Greys")
        pylab.show()
