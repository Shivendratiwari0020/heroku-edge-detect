import React, { useState, useEffect } from "react";
import "./modeling.css";
import Typography from "@mui/material/Typography";
import Player from "../../../assets/images/player.png";
import Setting from "../../../assets/images/settings.png";
import Parameters from "../../../assets/images/parameters.png";
import axios from "axios";
import DataTable from "../../datatable"
import { Grid } from "@mui/material";


  

function Modeling() {
  const [myData, setData] = useState([]);
  const [dataSummary, setDataSummary] = useState([])
  const [dataSummary2, setDataSummary2] = useState([])


  const data =[
	{modelName:"XGBoost", label:"XGBoost"},
	  {modelName:"Logistic Regression", label:"Logistic Regression"},
	 { modelName:"CATBOOST", label:"CATBOOST"},
	{modelName: "Random Forest", label:"Random Forest"},
	 {modelName:"Decision Tree", label:"Decision Tree"},
	{modelName:"Gradient Boosting Machines", label:"XGBoost"},
 ];
 const [options] = useState(data)

  // const getData = () => {

  // 		axios.get("https://jsonplaceholder.typicode.com/posts")
  // 	.then((response) => {
  // 		console.log(response.data);
  // 		setData(response.data.body);
  // 	})
  // 	.catch ((error) => {
  // 		console.log(error)
  // 		});

  // };
  useEffect(()=> 	axios.get("http://127.0.0.1:8000/api/modeling")
 .then((response) => {
	 console.log(response.data.data);
	 setData(response.data.data);
	 console.log(response.data.summary);
	 setDataSummary(response.data.NumericalDataSummary);
   setDataSummary2(response.data.CategoricalDataSummary);

 })
 .catch ((error) => {
	 console.log(error)
	 }),
	  []);

  const [models, setModels] = useState({
    targetVariable: "",
    splitPercentage: "",
    sampling: "",
	modelData1:[]
  });

  function handle(e) {
	let selectElement = document.getElementById('newModels')
	let modelValues = Array.from(selectElement.selectedOptions)
			.map(option => option.value)
    models.modelData1= modelValues
    let modelData = { ...models,};
    modelData[e.target.id] = e.target.value;
    console.log(modelData);
    setModels(modelData);
  }
  function clicked(e) {
    e.preventDefault();
    axios
      .post("http://127.0.0.1:8000/api/startmodeling", {
		targetVariable: models.targetVariable,
		splitPercentage: models.splitPercentage,
		sampling: models.sampling,
		modelData1:models.modelData1
      },)
      .then((res) => {
        console.log(res.data);
      });
	}




  return (
    <>
	<Grid>
      <Grid class="sub-nav">
        <Grid class="subnav-modeling-content">
          <a href="/Modeling">SUPERVISED</a>
          <a href="">UNSUPERVISED</a>
          <a href="">TIME SERIES</a>
        </Grid>
      </Grid>
	  <Grid className="visualData">
      <Grid className="dataDisplay1">
        <Grid class="target-selector">
          <label for="target-dropdown">Choose Target Variable: </label>
          <select
	
		 
            id="targetVariable"
            value={models.targetVariable.value}
            name="target-dropdown"
            onChange={(e) => handle(e)}
          >
        {myData.map(newOption =>  (<option value={newOption.userId} key={newOption}>{newOption.userId}</option>))}
            {/* <option value="1">Var1</option>
		    <option value="2">Var2</option>
		    <option value="3">Var3</option>
		    <option value="4">Var4</option>
		    <option value="5">Var5</option>
		    <option value="6">Var6</option>
		    <option value="7">Var7</option>
		    <option value="8">Var8</option>
		    <option value="9">Var9</option>
		    <option value="10">Var10</option>
		    <option value="11">Var11</option>
		    <option value="12">Var12</option> */}
          </select>
        </Grid>

        <Grid class="target-selector-2">
          <label for="target-dropdown-2">Choose Models: </label>
           <select
            id="newModels"
		    multiple 
            className="ui fluid dropdown"
            name="target-dropdown-2"
            onChange={(e) => handle(e)}			
          >
            <option value="XGBoost">XGBoost</option>
            <option value="Logistic Regression">Logistic Regression</option>
            <option value="CATBOOST">CATBOOST</option>
            <option value="Random Forest">Random Forest</option>
            <option value="Decision Tree">Decision Tree</option>
            <option value="Gradient Boosting Machines">
              Gradient Boosting Machines
            </option>   
           </select> 
          
		  
        </Grid>
        <Grid class="taget-selector-3">
          <label for="split">Enter Test Split Percentage: </label>
          <input
            type="text"
            id="splitPercentage"
            value={models.splitPercentage}
            placeholder="Enter Test Split Percentage"
            onChange={(e) => handle(e)}
			
          ></input>
        </Grid>
        <Grid class="taget-selector-4">
          <label for="Sampling">Undersample/Downsample: </label>
          <select
            id="sampling"
            value={models.sampling}
            onChange={(e) => handle(e)}
			maxLength="11"
          >
            <option value="Undersample">Undersample</option>
            <option value="Oversample">Oversample</option>
            <option value="None">None</option>
          </select>
        </Grid>
      </Grid>
      <Grid className="modelingSystem">
        <Grid className="startModeling">
          <Grid className="modelstartimage">
            <img src={Player} alt="ModelstartImage" onClick={clicked}></img>
            <Typography>Start Modeling</Typography>
          </Grid>
          <Grid className="setting">
            <Grid className="settingImage">
              <img src={Setting} alt="ModelstartImage" onClick={clicked}></img>
              <Typography>Modeling Setting</Typography>
            </Grid>
            <Grid className="parameters">
              <img
                src={Parameters}
                alt="ModelstartImage"
                onClick={clicked}
              ></img>
              <Typography align="center">Hyper Parameters</Typography>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
	  </Grid>
      <Grid className="dataSummary">
		  <Grid className="dataText">
        <Typography className="dataSummaryText">Numerical Data Summary</Typography>
        </Grid>
          
          <Grid className="dataSummaryData">
           <DataTable  data={dataSummary} />
        </Grid>
        <Grid>
      <Grid id ="categoricalData" className="dataText">
        <Typography className="dataSummaryText">Categorical Data Summary</Typography>
        </Grid>
        <DataTable  data={dataSummary2} />
      </Grid>
      </Grid>
    
	  </Grid>
    </>
  );
}

export default Modeling;
