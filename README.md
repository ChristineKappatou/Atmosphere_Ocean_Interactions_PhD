This GitHub Repository is connected to both Local and my NIRD servers. 

For this, I created an SSH key for ach server.

To add the GitHub extention on Jupyter Lab in the correct environment :

1. pip install jupyterlab-git
2. jupyter serverextension enable --py jupyterlab_git

To "see" your folder in the Git Environment of the extention and then push to the repository:

1. git init #Initialize git in your notebook folder
2. git add your_notebook.ipynb #add your notebook to the "staged" group
3. git commit -m "Initial commit" # commit the change

4. git branch -M main # start connecting to the github repository
5. git remote add origin https://github.com/USERNAME/REPO_NAME.git
6. git push -u origin main

7. git add your_notebook.ipynb # sync your changes with GitHub
8. git commit -m "Updated analysis"
9. git push

The steps you follow to access your NIRD account and files:

1. Create a user account at https://www.metacenter.no/user/application/form/storage/, having previously communicated with a person that has a project there and can accept your request.

2. Open a terminal window and ssh conect to @login.nird.sigma2.no

3. Type 'module load Miniforge3/24.1.2-0'

4. Type 'source activate' (onda activate/deactivate will not work unless you do 3, 4)

5. Type 'conda activate your_environment'

6. Type 'jupyter lab --port=8080' (jupyter lab has to be installed in the environment first)

This notebook contains:

Files from Local

1. Long_CMIP6.ipynb: Ibmalance and AHT, OHT for NorESM2-LM (500 years) and IPSL-CM6A-LR (900 years)
2. long_timescale_cesm2.ipynb: Ibmalance and AHT, OHT for CESM2 (999 years)

Files from NIRD:

1. AMOC_NIRD: AMOC and Deacon's cell calculations for all 3 models

8. Open a new terminal window (so you're on local) and start listening to the port on nird: 'ssh -L 8080:localhost:8080 username@loginX.nird.sigma2.no',  X is the number of the login node you run jupyter on.

9. Open the link from the nird window in browser or just do 'localhost:8080' in your browser.
