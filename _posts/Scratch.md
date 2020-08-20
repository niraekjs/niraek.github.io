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
df
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
    <tr>
      <th>5</th>
      <td>4.0368</td>
      <td>52.0</td>
      <td>4.761658</td>
      <td>1.103627</td>
      <td>413.0</td>
      <td>2.139896</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>2.697</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.6591</td>
      <td>52.0</td>
      <td>4.931907</td>
      <td>0.951362</td>
      <td>1094.0</td>
      <td>2.128405</td>
      <td>37.84</td>
      <td>-122.25</td>
      <td>2.992</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.1200</td>
      <td>52.0</td>
      <td>4.797527</td>
      <td>1.061824</td>
      <td>1157.0</td>
      <td>1.788253</td>
      <td>37.84</td>
      <td>-122.25</td>
      <td>2.414</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0804</td>
      <td>42.0</td>
      <td>4.294118</td>
      <td>1.117647</td>
      <td>1206.0</td>
      <td>2.026891</td>
      <td>37.84</td>
      <td>-122.26</td>
      <td>2.267</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.6912</td>
      <td>52.0</td>
      <td>4.970588</td>
      <td>0.990196</td>
      <td>1551.0</td>
      <td>2.172269</td>
      <td>37.84</td>
      <td>-122.25</td>
      <td>2.611</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.2031</td>
      <td>52.0</td>
      <td>5.477612</td>
      <td>1.079602</td>
      <td>910.0</td>
      <td>2.263682</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>2.815</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.2705</td>
      <td>52.0</td>
      <td>4.772480</td>
      <td>1.024523</td>
      <td>1504.0</td>
      <td>2.049046</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>2.418</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.0750</td>
      <td>52.0</td>
      <td>5.322650</td>
      <td>1.012821</td>
      <td>1098.0</td>
      <td>2.346154</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>2.135</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.6736</td>
      <td>52.0</td>
      <td>4.000000</td>
      <td>1.097701</td>
      <td>345.0</td>
      <td>1.982759</td>
      <td>37.84</td>
      <td>-122.26</td>
      <td>1.913</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.9167</td>
      <td>52.0</td>
      <td>4.262903</td>
      <td>1.009677</td>
      <td>1212.0</td>
      <td>1.954839</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>1.592</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.1250</td>
      <td>50.0</td>
      <td>4.242424</td>
      <td>1.071970</td>
      <td>697.0</td>
      <td>2.640152</td>
      <td>37.85</td>
      <td>-122.26</td>
      <td>1.400</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.7750</td>
      <td>52.0</td>
      <td>5.939577</td>
      <td>1.048338</td>
      <td>793.0</td>
      <td>2.395770</td>
      <td>37.85</td>
      <td>-122.27</td>
      <td>1.525</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.1202</td>
      <td>52.0</td>
      <td>4.052805</td>
      <td>0.966997</td>
      <td>648.0</td>
      <td>2.138614</td>
      <td>37.85</td>
      <td>-122.27</td>
      <td>1.555</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.9911</td>
      <td>50.0</td>
      <td>5.343675</td>
      <td>1.085919</td>
      <td>990.0</td>
      <td>2.362768</td>
      <td>37.84</td>
      <td>-122.26</td>
      <td>1.587</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.6033</td>
      <td>52.0</td>
      <td>5.465455</td>
      <td>1.083636</td>
      <td>690.0</td>
      <td>2.509091</td>
      <td>37.84</td>
      <td>-122.27</td>
      <td>1.629</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.3578</td>
      <td>40.0</td>
      <td>4.524096</td>
      <td>1.108434</td>
      <td>409.0</td>
      <td>2.463855</td>
      <td>37.85</td>
      <td>-122.27</td>
      <td>1.475</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.7135</td>
      <td>42.0</td>
      <td>4.478142</td>
      <td>1.002732</td>
      <td>929.0</td>
      <td>2.538251</td>
      <td>37.85</td>
      <td>-122.27</td>
      <td>1.598</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.7250</td>
      <td>52.0</td>
      <td>5.096234</td>
      <td>1.131799</td>
      <td>1015.0</td>
      <td>2.123431</td>
      <td>37.84</td>
      <td>-122.27</td>
      <td>1.139</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.1806</td>
      <td>52.0</td>
      <td>5.193846</td>
      <td>1.036923</td>
      <td>853.0</td>
      <td>2.624615</td>
      <td>37.84</td>
      <td>-122.27</td>
      <td>0.997</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2.6000</td>
      <td>52.0</td>
      <td>5.270142</td>
      <td>1.035545</td>
      <td>1006.0</td>
      <td>2.383886</td>
      <td>37.84</td>
      <td>-122.27</td>
      <td>1.326</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.4038</td>
      <td>41.0</td>
      <td>4.495798</td>
      <td>1.033613</td>
      <td>317.0</td>
      <td>2.663866</td>
      <td>37.85</td>
      <td>-122.28</td>
      <td>1.075</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.4597</td>
      <td>49.0</td>
      <td>4.728033</td>
      <td>1.020921</td>
      <td>607.0</td>
      <td>2.539749</td>
      <td>37.85</td>
      <td>-122.28</td>
      <td>0.938</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.8080</td>
      <td>52.0</td>
      <td>4.780856</td>
      <td>1.060453</td>
      <td>1102.0</td>
      <td>2.775819</td>
      <td>37.85</td>
      <td>-122.28</td>
      <td>1.055</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.6424</td>
      <td>50.0</td>
      <td>4.401691</td>
      <td>1.040169</td>
      <td>1131.0</td>
      <td>2.391121</td>
      <td>37.84</td>
      <td>-122.28</td>
      <td>1.089</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.6875</td>
      <td>52.0</td>
      <td>4.703226</td>
      <td>1.032258</td>
      <td>395.0</td>
      <td>2.548387</td>
      <td>37.84</td>
      <td>-122.28</td>
      <td>1.320</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20610</th>
      <td>1.3631</td>
      <td>28.0</td>
      <td>4.851936</td>
      <td>1.102506</td>
      <td>1195.0</td>
      <td>2.722096</td>
      <td>39.10</td>
      <td>-121.56</td>
      <td>0.455</td>
    </tr>
    <tr>
      <th>20611</th>
      <td>1.2857</td>
      <td>27.0</td>
      <td>4.359413</td>
      <td>1.078240</td>
      <td>1163.0</td>
      <td>2.843521</td>
      <td>39.10</td>
      <td>-121.55</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>20612</th>
      <td>1.4934</td>
      <td>26.0</td>
      <td>5.157303</td>
      <td>1.082397</td>
      <td>761.0</td>
      <td>2.850187</td>
      <td>39.08</td>
      <td>-121.56</td>
      <td>0.483</td>
    </tr>
    <tr>
      <th>20613</th>
      <td>1.4958</td>
      <td>31.0</td>
      <td>4.500000</td>
      <td>0.950521</td>
      <td>1167.0</td>
      <td>3.039062</td>
      <td>39.09</td>
      <td>-121.55</td>
      <td>0.534</td>
    </tr>
    <tr>
      <th>20614</th>
      <td>2.4695</td>
      <td>26.0</td>
      <td>4.801688</td>
      <td>0.970464</td>
      <td>1455.0</td>
      <td>3.069620</td>
      <td>39.08</td>
      <td>-121.54</td>
      <td>0.580</td>
    </tr>
    <tr>
      <th>20615</th>
      <td>2.3598</td>
      <td>23.0</td>
      <td>5.461929</td>
      <td>1.096447</td>
      <td>724.0</td>
      <td>3.675127</td>
      <td>39.08</td>
      <td>-121.54</td>
      <td>0.575</td>
    </tr>
    <tr>
      <th>20616</th>
      <td>2.0469</td>
      <td>15.0</td>
      <td>4.826667</td>
      <td>1.176000</td>
      <td>1157.0</td>
      <td>3.085333</td>
      <td>39.08</td>
      <td>-121.53</td>
      <td>0.551</td>
    </tr>
    <tr>
      <th>20617</th>
      <td>3.3021</td>
      <td>20.0</td>
      <td>4.921053</td>
      <td>0.956140</td>
      <td>308.0</td>
      <td>2.701754</td>
      <td>39.06</td>
      <td>-121.53</td>
      <td>0.708</td>
    </tr>
    <tr>
      <th>20618</th>
      <td>2.2500</td>
      <td>25.0</td>
      <td>5.893805</td>
      <td>1.092920</td>
      <td>726.0</td>
      <td>3.212389</td>
      <td>39.06</td>
      <td>-121.55</td>
      <td>0.634</td>
    </tr>
    <tr>
      <th>20619</th>
      <td>2.7303</td>
      <td>22.0</td>
      <td>6.388514</td>
      <td>1.148649</td>
      <td>1023.0</td>
      <td>3.456081</td>
      <td>39.01</td>
      <td>-121.56</td>
      <td>0.991</td>
    </tr>
    <tr>
      <th>20620</th>
      <td>4.5625</td>
      <td>40.0</td>
      <td>4.125000</td>
      <td>0.854167</td>
      <td>151.0</td>
      <td>3.145833</td>
      <td>39.05</td>
      <td>-121.48</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>20621</th>
      <td>2.3661</td>
      <td>37.0</td>
      <td>7.923567</td>
      <td>1.573248</td>
      <td>484.0</td>
      <td>3.082803</td>
      <td>39.01</td>
      <td>-121.47</td>
      <td>0.775</td>
    </tr>
    <tr>
      <th>20622</th>
      <td>2.4167</td>
      <td>20.0</td>
      <td>4.808917</td>
      <td>0.936306</td>
      <td>457.0</td>
      <td>2.910828</td>
      <td>39.00</td>
      <td>-121.44</td>
      <td>0.670</td>
    </tr>
    <tr>
      <th>20623</th>
      <td>2.8235</td>
      <td>32.0</td>
      <td>5.101322</td>
      <td>1.074890</td>
      <td>598.0</td>
      <td>2.634361</td>
      <td>39.03</td>
      <td>-121.37</td>
      <td>0.655</td>
    </tr>
    <tr>
      <th>20624</th>
      <td>3.0739</td>
      <td>16.0</td>
      <td>5.835052</td>
      <td>1.030928</td>
      <td>731.0</td>
      <td>2.512027</td>
      <td>39.04</td>
      <td>-121.41</td>
      <td>0.872</td>
    </tr>
    <tr>
      <th>20625</th>
      <td>4.1250</td>
      <td>37.0</td>
      <td>7.285714</td>
      <td>1.214286</td>
      <td>29.0</td>
      <td>2.071429</td>
      <td>39.12</td>
      <td>-121.52</td>
      <td>0.720</td>
    </tr>
    <tr>
      <th>20626</th>
      <td>2.1667</td>
      <td>36.0</td>
      <td>6.573099</td>
      <td>1.076023</td>
      <td>504.0</td>
      <td>2.947368</td>
      <td>39.18</td>
      <td>-121.43</td>
      <td>0.938</td>
    </tr>
    <tr>
      <th>20627</th>
      <td>3.0000</td>
      <td>5.0</td>
      <td>6.067797</td>
      <td>1.101695</td>
      <td>169.0</td>
      <td>2.864407</td>
      <td>39.13</td>
      <td>-121.32</td>
      <td>1.625</td>
    </tr>
    <tr>
      <th>20628</th>
      <td>2.5952</td>
      <td>19.0</td>
      <td>5.238462</td>
      <td>1.079487</td>
      <td>1018.0</td>
      <td>2.610256</td>
      <td>39.10</td>
      <td>-121.48</td>
      <td>0.924</td>
    </tr>
    <tr>
      <th>20629</th>
      <td>2.0943</td>
      <td>28.0</td>
      <td>5.519802</td>
      <td>1.020902</td>
      <td>6912.0</td>
      <td>3.801980</td>
      <td>39.12</td>
      <td>-121.39</td>
      <td>1.083</td>
    </tr>
    <tr>
      <th>20630</th>
      <td>3.5673</td>
      <td>11.0</td>
      <td>5.932584</td>
      <td>1.134831</td>
      <td>1257.0</td>
      <td>2.824719</td>
      <td>39.29</td>
      <td>-121.32</td>
      <td>1.120</td>
    </tr>
    <tr>
      <th>20631</th>
      <td>3.5179</td>
      <td>15.0</td>
      <td>6.145833</td>
      <td>1.141204</td>
      <td>1200.0</td>
      <td>2.777778</td>
      <td>39.33</td>
      <td>-121.40</td>
      <td>1.072</td>
    </tr>
    <tr>
      <th>20632</th>
      <td>3.1250</td>
      <td>15.0</td>
      <td>6.023377</td>
      <td>1.080519</td>
      <td>1047.0</td>
      <td>2.719481</td>
      <td>39.26</td>
      <td>-121.45</td>
      <td>1.156</td>
    </tr>
    <tr>
      <th>20633</th>
      <td>2.5495</td>
      <td>27.0</td>
      <td>5.445026</td>
      <td>1.078534</td>
      <td>1082.0</td>
      <td>2.832461</td>
      <td>39.19</td>
      <td>-121.53</td>
      <td>0.983</td>
    </tr>
    <tr>
      <th>20634</th>
      <td>3.7125</td>
      <td>28.0</td>
      <td>6.779070</td>
      <td>1.148256</td>
      <td>1041.0</td>
      <td>3.026163</td>
      <td>39.27</td>
      <td>-121.56</td>
      <td>1.168</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>1.5603</td>
      <td>25.0</td>
      <td>5.045455</td>
      <td>1.133333</td>
      <td>845.0</td>
      <td>2.560606</td>
      <td>39.48</td>
      <td>-121.09</td>
      <td>0.781</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>2.5568</td>
      <td>18.0</td>
      <td>6.114035</td>
      <td>1.315789</td>
      <td>356.0</td>
      <td>3.122807</td>
      <td>39.49</td>
      <td>-121.21</td>
      <td>0.771</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>1.7000</td>
      <td>17.0</td>
      <td>5.205543</td>
      <td>1.120092</td>
      <td>1007.0</td>
      <td>2.325635</td>
      <td>39.43</td>
      <td>-121.22</td>
      <td>0.923</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>1.8672</td>
      <td>18.0</td>
      <td>5.329513</td>
      <td>1.171920</td>
      <td>741.0</td>
      <td>2.123209</td>
      <td>39.43</td>
      <td>-121.32</td>
      <td>0.847</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>2.3886</td>
      <td>16.0</td>
      <td>5.254717</td>
      <td>1.162264</td>
      <td>1387.0</td>
      <td>2.616981</td>
      <td>39.37</td>
      <td>-121.24</td>
      <td>0.894</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 9 columns</p>
</div>

