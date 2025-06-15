# Earthquake-Here we downloaded the earthquake data from kaggle.
Here we got information about the columns in the dataset.
time:Date and time the earthquake occured.
latitude:Latitude of the earthquake epicenter.
longitude:Longitude of the earthquake epicenter.
depth:Depth of the occured earthquke in the place of its epicenter.
mag:Magnitude of the earthquake occured.
magType:Type of magnitude(md-Duration magnitude,ml-local magnitude,mb-body wave magnitude,ms-surface wave magnitude,mw-moment magnitude).
nst-Number of seismic stations that reported the event.
gap-Azimuthal gap (in degrees) between the closest stations.
dmin-Minimum distance to the nearest station (degrees).
rms-Root mean square in wave amplitude.If its high it represent that there is high signal or inaccurate reading of measurment.
net-Network that provides the data.
id-Unique identifier of the earthquake occured.
updates-Date and time when the earthqauke occured was last updated.
place-Place where the earthquake occured so it can be easily identified by us.
type-Type of natural disaster that occured(earthqauke,explosion).
horizontalError-Horiontal location error.
depthError-depth location error.
magError-error in magnitude measuremnt.
magNst-Number of stations used to calculate magnitude.
status-	Status of the event (reviewed, automatic).
locationSource-	Source of location data.
magSource-	Source of magnitude data.
First we are finding the duplicate and missing value and deleting it.
Then we are processing with the outliers.If it have higher outliers value,we are deleting it.
Then we are performaning univariate,bivariate,multivariate correlation analysis.
During multivariate correaltion analysis if its value is more than 5 we are removing it.
Then we are finding kmean cluster and silhouette score.
Then we are training and testing the data using models like linear regression,DecisionTree regression,GradientBoostingRegressor,logistic regression.
We have also performed decision tree.
