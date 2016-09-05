# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:39:09 2016

@author: rutvij
"""
def readFile(path):
    with open(path) as f:
        lines = f.readlines()
    content = [x.strip('\n') for x in lines]
    return content

input=readFile("/home/rutvij/workspace/Coursera/Coursera/test")

def countInv(input):
    mid = len(input)/2
    leftInv =0
    midInv=0
    rightInv=0
    
    if(len(input)<=1):
        return 0
        
    leftList = input[:mid]
    rightList = input[mid:]
    
    leftInv=countInv(leftList)
    rightInv=countInv(rightList)
    result = []
    midInv=mergeandsort(leftList,rightList,result)
    input[:] = result[:]
    return leftInv + rightInv + midInv
    
def mergeandsort(leftList,rightList,result):
    a=0
    b=0
    inv=0
    while(a<len(leftList) and b<len(rightList)):
        if(leftList[a]<=rightList[b]):
            result.append(leftList[a])
            a=a+1
           
        else:
            result.append(rightList[b])
            inv = inv + len(leftList) - a
            b=b+1
            
    while(a<len(leftList)):
        result.append(leftList[a])
        a=a+1
        
    while(b<len(rightList)):
        result.append(rightList[b])
        b=b+1
    
    return inv
        
    
print(countInv(input))
#print(input)    