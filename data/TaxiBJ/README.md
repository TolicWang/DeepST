TaxiBJ: InFlow/OutFlow, Meteorology and Holidays at Beijing
===========================================================

**If you use the data, please cite the following paper.**

`Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017. `

Download data from [OneDrive](https://1drv.ms/f/s!Akh6N7xv3uVmhOhDKwx3bm5zpHkDOQ) or [BaiduYun](http://pan.baidu.com/s/1qYq7ja8)

Please check the data with `md5sum` command: 
```
md5sum -c md5sum.txt
```

**TaxiBJ** consists of the following **SIX** datasets:

* BJ16_M32x32_T30_InOut.h5
* BJ15_M32x32_T30_InOut.h5
* BJ14_M32x32_T30_InOut.h5
* BJ13_M32x32_T30_InOut.h5
* BJ_Meteorology.h5
* BJ_Holiday.txt

where the first four files are *crowd flows* in Beijing from the year 2013 to 2016, `BJ_Meteorology.h5` is the Meteorological data, `BJ_Holiday.txt` includes the holidays (and adjacent weekends) of Beijing. 

Note: `*.h5` is `hdf5` file, one can use the follow code to view the data:

```
import h5py
f = h5py.File('BJ16_M32x32_T30_InOut.h5')
for ke in f.keys():
    print(ke, f[ke].shape)
```

## Flows of Crowds

File names: `BJ[YEAR]_M32x32_T30_InOut.h5`, where

* YEAR: one of {13, 14, 15, 16}
* M32x32: the Beijing city is divided into a 32 x 32 grid map
* T30: timeslot (a.k.a. time interval) is equal to 30 minites, meaning there are 48 timeslots in a day
* InOut: Inflow/Outflow are defined in the following paper [1]. 

[1] Junbo Zhang, Yu Zheng, Dekang Qi. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction. In AAAI 2017. 

Each `*.h5` file has two following subsets:

* `date`: a list of timeslots, which is associated the **data**. 
* `data`: a 4D tensor of shape (number_of_timeslots, 2, 32, 32), of which `data[i]` is a 3D tensor of shape (2, 32, 32) at the timeslot `date[i]`, `data[i][0]` is a `32x32` inflow matrix and `data[i][1]` is a `32x32` outflow matrix. 

### Example

You can get the data info with following command: 
```
python -c "from deepst.datasets import stat; stat('BJ16_M32x32_T30_InOut.h5')"
```

The output looks like: 
```
=====stat=====
data shape: (7220, 2, 32, 32)
# of days: 162, from 2015-11-01 to 2016-04-10
# of timeslots: 7776
# of timeslots (available): 7220
missing ratio of timeslots: 7.2%
max: 1250.000, min: 0.000
=====stat=====
```

## Meteorology

File name: `BJ_Meteorology.h5`, which has four following subsets:

* `date`: a list of timeslots, which is associated the following kinds of data. 
* `Temperature`: a list of continuous value, of which the `i^{th}` value is `temperature` at the timeslot `date[i]`.
* `WindSpeed`: a list of continuous value, of which the `i^{th}` value is `wind speed` at the timeslot `date[i]`. 
* `Weather`: a 2D matrix, each of which is a one-hot vector (`dim=17`), showing one of the following weather types: 
```
Sunny = 0,  
Cloudy = 1, 
Overcast = 2, 
Rainy = 3, 
Sprinkle = 4,  
ModerateRain = 5,  
HeavyRain = 6, 
Rainstorm = 7, 
Thunderstorm = 8, 
FreezingRain = 9, 
Snowy = 10,  
LightSnow = 11, 
ModerateSnow = 12, 
HeavySnow = 13, 
Foggy = 14,  
Sandstorm = 15, 
Dusty = 16, 
```

## Holiday

File name: `BJ_Holiday.txt`, which inclues a list of the holidays (and adjacent weekends) of Beijing. 

Each line a holiday with the data format [yyyy][mm][dd]. For example, `20150601` is `June 1st, 2015`. 