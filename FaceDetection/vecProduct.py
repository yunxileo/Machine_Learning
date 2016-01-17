"""
Programmer  :   EOF
Date        :   2016.01.16
File        :   vecProduct.py
E-mail      :   jasonleaster@163.com

"""
def VecProduct(Vec1, Vec2):
    assert len(Vec1) == len(Vec2)
    summer = 0.
    for i, j in zip(Vec1, Vec2):
        summer += i * j

    return summer

