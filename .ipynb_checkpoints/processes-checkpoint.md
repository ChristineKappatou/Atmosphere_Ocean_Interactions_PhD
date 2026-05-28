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

*if a token is asked, go to the NIRD tab on terminal and copy paste what comes after this:  http://localhost:8080/lab?token=...


## <u>Updating NIRD with the data you download:</u>

1. cd /datalake/NS9560K/ESGF/rawdata/model

2. mkdir: folder with your name if it doesn't exist to dump the data into

3. cd /datalake/NS9560K/ESGF/rawdata

4. ./move2autosort.sh "/path"

*might take some time, plus it renews every half an hour

## <u>Download selected files from an .nc file:</u>

1. Download the whole .wget file
2. Open a teminal window and cd in the folder where you downloaded the file
3. chmod +x <wget_file_name> to make it executable
4. nano <wget_file_name> to get into the .nc file
5. remove the files with the years that are not relevant with ctrl+K and exit with ctrl + X + y to the saving question
6. ./<wget_file_name> to download the file. 

------------

# **GitHub**

## <u>Update remote:</u>

1. git add + file (for selected files) or git add . for everything you have changed
2. git commit -m "name_of_change"
3. git push + GitHub user name (i.e. ChristineKappatou) + GitHub password. 

* This requires that you have cloned the repo on local, you're in this folder making the changed and then commit while still in it. 

## <u>Update local:</u>  

git pull

*I prefer working from local ad updating the remote repo instead. 
