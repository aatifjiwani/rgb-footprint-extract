import math
import utm

X_MIN, Y_MIN = 3340084, 1970605
SPATIAL_RESOLUTIOn = 0.037 # in m, level 22

def xy2lonlat(x, y, z=22):
    lon = x / pow(2.0, z) * 360.0 - 180
    
    n = math.pi - (2.0 * math.pi * y) / pow(2.0, z)
    lat = math.degrees(math.atan(math.sinh(n)))

    return lon, lat


if __name__ == "__main__":
    lon, lat = xy2lonlat(X_MIN, Y_MIN, z=22)
    u = utm.from_latlon(lat, lon)
    print(u)