# Title



```python
import sklearn.datasets
import pandas as pd
import numpy as np
data = sklearn.datasets.fetch_california_housing()
df = pd.DataFrame(np.concatenate((data.data, np.array(
    [data.target]).T), axis=1), columns=data.feature_names + ['Med House Value'])
```

```python
#Linear Regression
def get_Y(df, y_name):
    return df.loc[:, y_name]
def get_X(df, y_name):
    return df.drop(y_name, axis = 1)
```

```python
df[["MedInc", "Med House Value"]]
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
      <th>Med House Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>3.422</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0368</td>
      <td>2.697</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.6591</td>
      <td>2.992</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.1200</td>
      <td>2.414</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0804</td>
      <td>2.267</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.6912</td>
      <td>2.611</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.2031</td>
      <td>2.815</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.2705</td>
      <td>2.418</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.0750</td>
      <td>2.135</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.6736</td>
      <td>1.913</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.9167</td>
      <td>1.592</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.1250</td>
      <td>1.400</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.7750</td>
      <td>1.525</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.1202</td>
      <td>1.555</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.9911</td>
      <td>1.587</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.6033</td>
      <td>1.629</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.3578</td>
      <td>1.475</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.7135</td>
      <td>1.598</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.7250</td>
      <td>1.139</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.1806</td>
      <td>0.997</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2.6000</td>
      <td>1.326</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.4038</td>
      <td>1.075</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.4597</td>
      <td>0.938</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.8080</td>
      <td>1.055</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.6424</td>
      <td>1.089</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.6875</td>
      <td>1.320</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20610</th>
      <td>1.3631</td>
      <td>0.455</td>
    </tr>
    <tr>
      <th>20611</th>
      <td>1.2857</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>20612</th>
      <td>1.4934</td>
      <td>0.483</td>
    </tr>
    <tr>
      <th>20613</th>
      <td>1.4958</td>
      <td>0.534</td>
    </tr>
    <tr>
      <th>20614</th>
      <td>2.4695</td>
      <td>0.580</td>
    </tr>
    <tr>
      <th>20615</th>
      <td>2.3598</td>
      <td>0.575</td>
    </tr>
    <tr>
      <th>20616</th>
      <td>2.0469</td>
      <td>0.551</td>
    </tr>
    <tr>
      <th>20617</th>
      <td>3.3021</td>
      <td>0.708</td>
    </tr>
    <tr>
      <th>20618</th>
      <td>2.2500</td>
      <td>0.634</td>
    </tr>
    <tr>
      <th>20619</th>
      <td>2.7303</td>
      <td>0.991</td>
    </tr>
    <tr>
      <th>20620</th>
      <td>4.5625</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>20621</th>
      <td>2.3661</td>
      <td>0.775</td>
    </tr>
    <tr>
      <th>20622</th>
      <td>2.4167</td>
      <td>0.670</td>
    </tr>
    <tr>
      <th>20623</th>
      <td>2.8235</td>
      <td>0.655</td>
    </tr>
    <tr>
      <th>20624</th>
      <td>3.0739</td>
      <td>0.872</td>
    </tr>
    <tr>
      <th>20625</th>
      <td>4.1250</td>
      <td>0.720</td>
    </tr>
    <tr>
      <th>20626</th>
      <td>2.1667</td>
      <td>0.938</td>
    </tr>
    <tr>
      <th>20627</th>
      <td>3.0000</td>
      <td>1.625</td>
    </tr>
    <tr>
      <th>20628</th>
      <td>2.5952</td>
      <td>0.924</td>
    </tr>
    <tr>
      <th>20629</th>
      <td>2.0943</td>
      <td>1.083</td>
    </tr>
    <tr>
      <th>20630</th>
      <td>3.5673</td>
      <td>1.120</td>
    </tr>
    <tr>
      <th>20631</th>
      <td>3.5179</td>
      <td>1.072</td>
    </tr>
    <tr>
      <th>20632</th>
      <td>3.1250</td>
      <td>1.156</td>
    </tr>
    <tr>
      <th>20633</th>
      <td>2.5495</td>
      <td>0.983</td>
    </tr>
    <tr>
      <th>20634</th>
      <td>3.7125</td>
      <td>1.168</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>1.5603</td>
      <td>0.781</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>2.5568</td>
      <td>0.771</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>1.7000</td>
      <td>0.923</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>1.8672</td>
      <td>0.847</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>2.3886</td>
      <td>0.894</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 2 columns</p>
</div>


