import { Grid } from '@mui/material';
import React, { useState, useEffect} from 'react';
import axios from "axios";
import DataTable from "../../datatable"
import Typography from "@mui/material/Typography";
import "./data.css"


const Data = () => {
  const [numericalData, setNumericalData] = useState([])
  const [categoryData, setCategoryData] = useState([])





  useEffect(() => axios.get("http://127.0.0.1:8000/api/modeling")
  .then((response) => {
    console.log(response.data.data);
    setNumericalData(response.data.NumericalDataSummary);
    setCategoryData(response.data.CategoricalDataSummary);


  })
  .catch((error) => {
    console.log(error)
  }),
  []);
  return (
    <div>
       <Grid className="dataSummary">
          <Grid className="dataText">
            <Typography className="dataSummaryText">Continuous Data Summary</Typography>
          </Grid>

          <Grid className="dataSummaryData">
            <DataTable className="dataNumerical" data={numericalData} />



          </Grid>
          <Grid>
            <Grid id="categoricalData" className="dataText">
              <Typography className="dataSummaryText"> Missing Data Summary</Typography>
            </Grid>
            <Grid className="catData">
              <DataTable data={categoryData} />
            </Grid>
          </Grid>
        </Grid>
    </div>
  );
};

export default Data;