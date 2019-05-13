#!/usr/bin/env python3

from EasyTemikaXML import EasyTemikaXML

tilt=[8.4/6965.0,20.0/9943.0]
tilt=[(1828.8-1827.175)/2000,(1826.6-1824.9)/3000]
#tilt=[0,0]
xml=EasyTemikaXML(True, 1828.950)  #absolute_z

xml.rgb_image("/home/fa344/data/rgb_cal_new/cal1", [0.07,0.15,0.2])
xml.sleep(1)
xml.rgb_image("/home/fa344/data/rgb_cal_new/cal2", [0.07,0.15,0.2])
xml.sleep(1)
xml.rgb_image("/home/fa344/data/rgb_cal_new/cal3", [0.07,0.15,0.2])

#rgb=xml.rgb_image("/home/fa344/data/zstack_grid3/", [0.07,0.15,0.2],True)
#zs=xml.z_stack(3, 41, '', rgb, True)
#xml.image_grid([150,200],[6,6], zs, tilt)
xml.closing()
