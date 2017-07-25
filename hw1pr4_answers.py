#
# starting examples for cs35, week2 "Web as Input"
#

import requests
import string
import json

# CODE FOR FIND CITY BASED ON LAT/LONG COORDINATES
from bs4 import BeautifulSoup

def findAddress(latlngList):
  """prints addresses based on latitude and longitude coordinates"""
  latlngList = ['25.762233,-80.936365','35.424868,-92.724609', '45.166547,-98.739624' ,'35.150675,-116.106111','46.064581,-118.343021']
  for latlng in latlngList:
      
    # latlng = '25.762233,-80.936365'     Unnamed Road, Ochopee, FL 34141, USA
    # latlng = '35.424868,-92.724609'     267 Copeland Rd, Cleveland, AR 72030, USA
    # latlng = '45.166547,-98.739624'      2540-3006 3rd Ave E, Chelsea, SD 57465, USA
    # latlng = '35.150675,-116.106111'    Zzyzx Rd, Baker, CA 92309, USA
    # latlng = '46.064581,-118.343021'    S 5th Ave, Walla Walla, WA 99362, USA

    url = 'http://maps.googleapis.com/maps/api/geocode/xml?latlng='+latlng+'&sensor=true'


    response = requests.get(url)
    data = response.text
    soup = BeautifulSoup(data, "lxml")

    distance = soup.findAll('formatted_address')
    print(distance[0].text)
  