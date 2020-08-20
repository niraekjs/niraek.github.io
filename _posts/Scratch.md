# Title



Our end goal is to learn how simple linear regression works, and to implement it on a real dataset. First, let's take a look at an example dataset and go through the logic of linear regression on it.

The dataset we'll look at is of California housing data, from the 1990 US census. This dataset is automatically included in sklearn, which has many datasets you can explore. Below is the code to load this data, using Pandas dataframes, which you can think of as just fancy tables for now. We call the dataframe `ca_data`.

```python
import sklearn.datasets
import pandas as pd
import numpy as np
data = sklearn.datasets.fetch_california_housing()
ca_data = pd.DataFrame(np.concatenate((data.data, np.array(
    [data.target]).T), axis=1), columns=data.feature_names + ['Med House Value'])
```

Let's take a look at the first five rows of our table.

```python
ca_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Med House Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>



We see $9$ column labels. The leftmost column is called the index, which is just used to reference certain rows, and should not be considered a real column by itself. Reading the sklearn's documentation for the dataset, we see that each row represents "block group", i.e. groups delineated by the census containing about $1500$ people each. For instance, the first row is a block group that has a log median income of $8.3252$ (both MedInc and Med House Value are in log terms). 

What can we do with this dataset? A natural question is using this data, can we predict some characteristics of unseen data? For example, what if we wanted to predict housing value from income? We could make a model to do this - one of which is linear regression! This could be useful in many ways, for instance, real estate agents could ask potential buyers for their annual income, and consequently show them houses near the model's predicted housing value range. 

Let us take a look at how the median income versus hedian housing value data is distributed. We can do this easily by using `pandas.plot`, which uses the python library matplotlib.

```python
ca_data.plot.scatter("MedInc", "Med House Value")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff040002e48>




![png](Scratch_files/output_5_1.png)


```python
#Linear Regression
def get_Y(df, y_name):
    return df.loc[:, y_name]
def get_X(df, y_name):
    return df.drop(y_name, axis = 1)
```
