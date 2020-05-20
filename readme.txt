
Lite-AutoML

In a conda virtual environment, install the following-
sklearn
openml
tpot

Run conda install swig

Install auto-sklearn by
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

Lite-AutoML evaluates itself on OpenML datasets and the 3 frameworks
auto-sklearn
TPOT
hyperopt-sklearn

Run Lite-AutoML as-
python main.py <timeout_in_min> <output_filename> <openmlid>

