from __future__ import division
import numpy as np
import datetime
import pymysql
import pymysql.cursors
import matplotlib.pyplot as plt
from matplotlib._png import read_png
from operator import itemgetter

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
airports = ["SEA","PDX","FAR","BOI","BIL","SFO","LAX","ELP","DFW","IAH","PHX","LAS","DEN","MSP","MKE","ORD","STL","DTW","PIT","IAD","JFK","BOS","ATL","CLT","MIA"]
sqlsetairports = "(\""+'\", \"'.join(airports)+"\")"

## pull flights from the database
flights = sqlExec("SELECT Origin,Dest,Distance,AirTime FROM flights WHERE Origin IN "+sqlsetairports+" AND Dest IN "+sqlsetairports+" AND Month IN (\"1 2 12\") AND Cancelled = 0 AND Diverted = 0 AND UniqueCarrier != \"B6\"")
print "Total of "+str(len(flights))+" flights"

## mean speeds
vbarspeeds = []
## speed of sound at ~35000 ft
soundspeed = 660

## loop over airports, take all unordered pairs
for i in range(len(airports)):
   for j in range(i):
      ## get the origin and destination
      placeA = airports[i]
      placeB = airports[j]
      print "now working on "+placeA+" <-> "+placeB
      ## loop over flights
      speedsAtoB = []
      speedsBtoA = []
      for flight in flights:
         if flight["Origin"] == placeA and flight["Dest"] == placeB and flight["AirTime"] > 0:
            speedsAtoB.append(flight["Distance"]/(flight["AirTime"]/60.))
         elif flight["Origin"] == placeB and flight["Dest"] == placeA and flight["AirTime"] > 0:
            speedsBtoA.append(flight["Distance"]/(flight["AirTime"]/60.))
      if len(speedsAtoB) > 0 and len(speedsBtoA) > 0:
         ## make plot of speeds and save
         fig = plt.figure()
         ## set plot range, bins
         xmin = 0
         xmax = soundspeed * 1.1
         bins = np.linspace(xmin, xmax, 51)
         plt.xlim([xmin, xmax])
         plt.xlabel('speeds [mph]')
         ## histogram
         plt.hist(speedsAtoB, bins, facecolor='green', alpha=0.5, label=placeA+' to '+placeB)
         plt.hist(speedsBtoA, bins, facecolor='blue', alpha=0.5, label=placeB+' to '+placeA)
         plt.legend(loc='upper right')
         ## label the plot with the single event and its score
         ymax = 1.25*plt.axes().get_ylim()[1]
         plt.ylim([0,ymax])
         plt.axes().arrow(soundspeed, 0.12*ymax, 0, -0.09*ymax, head_width=0.015*(xmax - xmin), head_length=0.03*ymax, fc='r', ec='r')
         plt.savefig("plots/speeds_"+placeA+"_"+placeB+".pdf")
         plt.close(fig)

         ## calculate average speeds, offsets
         speedAtoB = np.mean(speedsAtoB)
         speedBtoA = np.mean(speedsBtoA)
         meanspeed = 0.5 * (speedAtoB + speedBtoA)
         vbar = 0.5 * (speedAtoB - speedBtoA)
         vbarspeeds.append([placeA,placeB,vbar])

outfile = open("average_speeds_DJF.dat","w")
outfile.write(str(vbarspeeds))
outfile.close()
