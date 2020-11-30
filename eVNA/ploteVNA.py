import numpy as np
import os
import Ngl, Nio
import scipy.stats as stats
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


year = 2017
mon = 1
day = 10
var = 'PM2.5'

data = np.load('eVNA_PM2.5_'+str(year)+'_'+str(mon)+'_d'+str(day)+'.npy')
print(data.shape)
xmean = np.mean(data)
MinF = 0
MaxF = 100
SpaceF = 10

f = Nio.open_file("/data1/home/dingd/dlrsm/emis/obs/day/VNA/GRIDCRO2D_cn27_2017001.nc","r")
lat2d = f.variables["LAT"][0,0,:,:]
lon2d = f.variables["LON"][0,0,:,:]
ncol = lat2d.shape[1]
nrow = lat2d.shape[0]
#print(ncol)
res = Ngl.Resources()
res.nglDraw = False
res.nglFrame = False
res.cnFillOn = True
res.cnLinesOn = False
res.cnLineLabelsOn = False
res.lbLabelBarOn = True
res.lbOrientation = "vertical"
#res.lbLabelStride = 2
res.cnInfoLabelOn = False
res.cnInfoLabelOrthogonalPosF = 0.04
res.cnInfoLabelString   = ""
res.sfXArray = lon2d[0,:]
res.sfYArray = lat2d[:,0]

res.mpProjection = "LambertConformal"
res.mpLambertParallel1F = 25.
res.mpLambertParallel2F = 40.
res.mpLambertMeridianF  = 110.
res.mpLimitMode   = "Corners"
res.mpLeftCornerLatF  = lat2d[0,0]
res.mpLeftCornerLonF  = lon2d[0,0]
res.mpRightCornerLatF = lat2d[nrow-1,ncol-1]
res.mpRightCornerLonF = lon2d[nrow-1,ncol-1]
res.cnConstFLabelFontHeightF = 0.0
res.tfDoNDCOverlay   = True
res.pmTickMarkDisplayMode = "Always"
res.tmXTOn = False
res.tmYROn = False
#res.tmXBOn = False
#res.tmYLOn = False

res.mpGeophysicalLineThicknessF = 3
res.mpOutlineBoundarySets   = "National"
res.mpOutlineSpecifiers   = ["China:states"]
res.mpDataSetName = "Earth..4"
res.mpDataBaseVersion = "MediumRes"
res.mpFillOn  = False
res.mpFillAreaSpecifiers  = ["China:states"]

res.tiMainFont= "helvetica"
res.tiMainOffsetYF= 0
res.tiMainFontHeightF = 0.02
res.tiMainPosition   = "Left"
res.tiMainFuncCode = "~"

res.mpGridAndLimbOn = True
res.mpGridLineDashPattern   = 2

res.pmTickMarkDisplayMode       = "Always"
res.lbOrientation       = "Vertical"
res.cnInfoLabelOn       = False
res.cnInfoLabelOrthogonalPosF = -0.04
res.cnInfoLabelString   = "Max: $ZMX$   Min: $ZMN$   Mean: "+str(xmean)
res.cnLevelSelectionMode= "ManualLevels"
res.cnFillPalette = "BlAqGrYeOrReVi200"
res.cnMinLevelValF      = MinF
res.cnMaxLevelValF      = MaxF
res.cnLevelSpacingF     = SpaceF
#res.lbLabelStride       = 10
res.vpXF                = 0.1
res.tiMainString        = "PM~B~2.5"
res.tiMainFontHeightF   = 0.028
res.tiDeltaF            = 1

wks = Ngl.open_wks("png","eVNA")
#-- create the contour plot
plot = Ngl.contour_map(wks,np.transpose(data),res)

#-- Retrieve some resources from map for adding labels
#vpx  = Ngl.get_float(plot.map,'vpXF')
#vpy  = Ngl.get_float(plot.map,'vpYF')
#vpw  = Ngl.get_float(plot.map,'vpWidthF')
#fnth = Ngl.get_float(plot.map,'tmXBLabelFontHeightF')

#-- write variable long_name and units to the plot
#txres               = Ngl.Resources()
#txres.txFontHeightF = fnth

#txres.txJust  = "CenterLeft"
#Ngl.text_ndc(wks,f.t.long_name,vpx,vpy+0.02,txres)
###Ngl.text_ndc(wks,f.variables["t"].attributes['long_name'],vpx,vpy+0.02,txres)

#txres.txJust  = "CenterRight"
#Ngl.text_ndc(wks,f.t.units,vpx+vpw,vpy+0.02,txres)
###Ngl.text_ndc(wks,f.variables["t"].attributes['units'],vpx,vpy+0.02,txres)

#-- advance the frame
Ngl.draw(plot)
Ngl.frame(wks)

Ngl.end()

#os.system("convert -density 300 -trim test.png test.png")
