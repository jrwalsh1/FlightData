from __future__ import division
import numpy as np
import numpy.linalg
import datetime
import pymysql
import pymysql.cursors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## program to use flight data from http://stat-computing.org/dataexpo/2009/
## to infer wind velocities in the range of the stratosphere where planes fly

## query the database
def sqlExec(query):
   """Given file name, reads input file and stores data"""
   db = pymysql.connect(user="root", host="localhost", passwd="", db="flight_data", cursorclass=pymysql.cursors.DictCursor)
   with db:
      cur = db.cursor()
      cur.execute(query)
      tables = cur.fetchall()
      return tables

## function to compute the geodesic distance (in miles) between two latitude, longitude pairs
def geodesicDist(lat1, long1, lat2, long2):
   """Computes geodesic distance between two latitude, longitude points, in miles.  Uses Earth radius = 3963 miles.  Assumes latitudes and longitudes are in degrees."""
   rEarth = 3963.
   ## convert latitudes to polar angles (radians)
   theta1 = 0.5 * np.pi - (np.pi / 180.) * lat1
   theta2 = 0.5 * np.pi - (np.pi / 180.) * lat2
   ## convert longitudes radians
   phi1 = (np.pi / 180.) * long1
   phi2 = (np.pi / 180.) * long2
   ## create unit vectors to each point
   unitvec1 = np.array([ np.sin(theta1) * np.cos(phi1), np.sin(theta1) * np.sin(phi1), np.cos(theta1) ])
   unitvec2 = np.array([ np.sin(theta2) * np.cos(phi2), np.sin(theta2) * np.sin(phi2), np.cos(theta2) ])
   ## calculate distance
   return rEarth * np.arccos( np.dot(unitvec1, unitvec2) )

## parametric function for a great circle between two longitude, latitude points
def greatCircleFunc(lat1, long1, lat2, long2, t):
   """Gives parametric formula for the great circle between two latitude, longitude points. Returns [latitude, longitude] in degrees."""
   ## convert latitudes to polar angles (radians)
   theta1 = 0.5 * np.pi - (np.pi / 180.) * lat1
   theta2 = 0.5 * np.pi - (np.pi / 180.) * lat2
   ## convert longitudes radians
   phi1 = (np.pi / 180.) * long1
   phi2 = (np.pi / 180.) * long2
   ## create unit vectors to each point
   unitvec1 = np.array([ np.sin(theta1) * np.cos(phi1), np.sin(theta1) * np.sin(phi1), np.cos(theta1) ])
   unitvec2 = np.array([ np.sin(theta2) * np.cos(phi2), np.sin(theta2) * np.sin(phi2), np.cos(theta2) ])
   ## angle between the two points
   tau = np.arccos( np.dot(unitvec1, unitvec2) )
   ## parametric terms
   A = np.sin((1 - t)*tau) / np.sin(tau)
   B = np.sin(t*tau) / np.sin(tau)
   ## compute latitude, longitude
   xt = A * np.sin(theta1) * np.cos(phi1) + B * np.sin(theta2) * np.cos(phi2)
   yt = A * np.sin(theta1) * np.sin(phi1) + B * np.sin(theta2) * np.sin(phi2)
   zt = A * np.cos(theta1) + B * np.cos(theta2)
   longt = (180. / np.pi) * np.arctan2(yt, xt)
   latt = (180. / np.pi) * np.arctan2(zt, np.sqrt(xt*xt + yt*yt))
   
   return [latt, longt]

## compute box fractions and path directions in latitude-longitude space
def boxfractions(lat1, long1, lat2, long2, boxes):
   """Computes the fraction of total distance of a geodesic path in a given latitude-longitude box.  Returns the distance fractions in the path for each box."""
   ## counter for each box
   boxpts = [0 for n in range(len(boxes))]
   ## loop over steps in t, use parametric function of geodesic path
   nsteps = 99
   for nstep in range(nsteps+1):
      ## get t value and latitude, longitude
      tval = nstep * 1. / nsteps
      [latt, longt] = greatCircleFunc(lat1, long1, lat2, long2, tval)
      ## loop over boxes
      for nbox in range(len(boxes)):
         ## check if our point is in the box
         if boxes[nbox][0][0] < latt < boxes[nbox][0][1] and boxes[nbox][1][0] < longt < boxes[nbox][1][1]:
            boxpts[nbox] += 1./(nsteps+1)
   return boxpts

## flights database
## Year, Month, DayofMonth, DayOfWeek, DepTime, CRSDepTime, ArrTime, CRSArrTime,
## UniqueCarrier, FlightNum, TailNum, ActualElapsedTime, CRSElapsedTime, AirTime,
## ArrDelay, DepDelay, Origin, Dest, Distance, TaxiIn, TaxiOut, Cancelled, CancellationCode,
## Diverted, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay

## carriers database
## Code, Description

## airports database
## Code, Airport, City, State, Country, Latitude, Longitude

## planes database
## TailNum, OwnerType, Manufacturer, IssueDate, Model, Status, AircraftType, EngineType, Year

## set of airports to loop over in pairs
airports = ["SEA","PDX","FAR","BOI","SFO","LAX","ELP","DFW","IAH","PHX","LAS","DEN","MSP","MKE","ORD","STL","DTW","PIT","IAD","JFK","BOS","ATL","CLT","MIA"]
sqlsetairports = "(\""+'\", \"'.join(airports)+"\")"
## get the latitude and longitude of each airport
airportrows = sqlExec("SELECT Code,Latitude,Longitude FROM airports WHERE Code IN "+sqlsetairports)
## put this into a dictionary
airportcoords = {}
for row in airportrows:
   airportcoords[row["Code"]] = [row["Latitude"], row["Longitude"]]

## latitude bins:
latitudebins = [[24, 32], [32, 36], [36, 40], [40, 44], [44, 50]]
## longitude bins
longitudebins = [[-124, -118], [-118, -110], [-110, -100], [-100, -92], [-92, -82], [-82, -76], [-76, -70]]

## get longitude-latitude boxes
## need to drop
## [-124, -118] x [24, 32]
## [-118, -110] x [24, 32]
## [-76, -70] x [24, 32]
## [-76, -70] x [32, 36]
## [-76, -70] x [44, 50]
boxes = []
boxcenters = []
for i in range(len(latitudebins)):
   for j in range(len(longitudebins)):
      ##      if (i == 0 and j < 3) or ((i < 2 or i == 4) and j == 6) or (i == 4 and (j == 2 or j == 5)):
      if (i == 0 and j < 3) or ((i < 2 or i == 4) and j == 6) or (i == 4 and j == 5):
         pass
      else:
         boxes.append([latitudebins[i], longitudebins[j]])
         boxcenters.append([0.5 * (latitudebins[i][0] + latitudebins[i][1]), 0.5 * (longitudebins[j][0] + longitudebins[j][1])])

## read the wind velocities for the flight paths from a saved data file
infile = open("average_speeds_DJF.dat","r")
vbarpaths = eval(infile.read())

## we are performing the least squares minimization for M.vJ = vbar, to find vJ
## where vJ are the components of wind speed velocity it latitude/longitude coordinates
## and vbar is the average wind speed for each path

## now construct M and vbar
## vbar
vbar = [vbarpath[2] for vbarpath in vbarpaths]

## M
rows = []
for path in vbarpaths:
   [lat1, long1] = airportcoords[path[0]]
   [lat2, long2] = airportcoords[path[1]]
   boxfracs = boxfractions(lat1, long1, lat2, long2, boxes)
   rhatlat = (lat2 - lat1) / np.sqrt((lat2 - lat1)*(lat2 - lat1) + (long2 - long1)*(long2 - long1))
   rhatlong = (long2 - long1) / np.sqrt((lat2 - lat1)*(lat2 - lat1) + (long2 - long1)*(long2 - long1))
   row = []
   for frac in boxfracs:
      row.append(rhatlat * frac)
      row.append(rhatlong * frac)
   rows.append(row)
M = np.matrix(rows)

## now find vJ
Mlsm = np.array(np.dot(np.linalg.inv(np.dot(np.transpose(M), M)),np.transpose(M)))
vJ = np.dot(Mlsm, vbar)

vJvals = []
for n in range(len(boxes)):
   vJvals.append(np.sqrt(vJ[2*n]*vJ[2*n] + vJ[2*n+1]*vJ[2*n+1]))
vJvals = np.sort(vJvals)
print vJvals

## US mercator map used here is 695-by-384
## coordinates are [0, 695] in longitude, [384, 0] in latitude
img=mpimg.imread('USmercator.png')

## show the backgronud
imgplot = plt.imshow(img)
plt.axes().xaxis.set_visible(False)
plt.axes().yaxis.set_visible(False)

## now add arrows for each box
for nbox in range(len(boxes)):
   ## convert the coordinates into the image frame
   ## longtitude (x): [-126, -66] -> [0, 695]
   ## latitude (x): [24, 50] -> [384, 0]
   [boxlat, boxlong] = boxcenters[nbox]
   boxlongcoord = 11.58333 * boxlong + 1459.5
   boxlatcoord = -14.7692 * boxlat + 738.462
   ## now get the velocity vector
   vJlat = vJ[2*nbox]
   vJlong = vJ[2*nbox + 1]
   vJpt = np.sqrt(vJlat*vJlat + vJlong*vJlong)
   rJlat = (vJlat / vJpt) * min((vJpt / vJvals[-2]), 1) * 50.
   rJlong = (vJlong / vJpt) * min((vJpt / vJvals[-2]), 1) * 50.
   ## add the arrow
   ## each box is ~ 75-by-75 pixels, so we normalize the arrow to have at most 50 pixels in length
   plt.axes().arrow(boxlongcoord, boxlatcoord, rJlong, rJlat, head_width=0.015*700, head_length=0.03*385, fc='black', ec='black')

plt.savefig("plots/windspeeds_DJF.pdf")
plt.show()

