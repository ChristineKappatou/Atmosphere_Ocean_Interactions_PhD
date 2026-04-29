# **NIRD**

## <u>Login on NIRD:</u>

On terminal:

1. ssh user@login.uio.no -> your UiO password
2. ssh chrikap@login.nird.sigma2.no -> your NIRD password 
3. module load Miniforge3/24.1.2-0
4. source activate
5. conda activate your_environment (nird_heat_transport in my case)
6. jupyter lab --port=8080 (jupyter lab has to be installed in the environment first)
7. Open a new tab on local and listen: ssh -L 8080:localhost:8080 chrikap@loginX.nird.sigma2.no, X is the number of the login node you run jupyter on
8. Open a new tab on browser: http://localhost:8080/ 
* if a token is asked, go to the NIRD tab on terminal and copy paste what comes after this:  http://localhost:8080/lab?token=...


Updating NIRD with the data you download:

1. cd /datalake/NS9560K/ESGF/rawdata/model

2. mkdir: folder with your name if it doesn't exist to dump the data into

3. cd /datalake/NS9560K/ESGF/rawdata

4. ./move2autosort.sh "/path"

*might take some time, plus it renews every half an hour

------------

# **GitHub**

## <u>Update remote:</u>

1. git add + file (for selected files) or git add . for everything you have changed
2. git commit -m "name_of_change"
3. git push + GitHub user name (i.e. ChristineKappatou) + GitHub password. 

* This requires that you have cloned the repo on local, you're in this folder making the changed and then commit while still in it. 

## <u>Update local:</u>  git pull

* I prefer working from local ad updating the remote repo instead. 
