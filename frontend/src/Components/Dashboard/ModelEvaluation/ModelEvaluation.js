import React, { useState, useEffect } from "react";
import './ModelEvaluation.css';
import Typography from '@mui/material/Typography';
import Grid from "@mui/material/Grid";
import axios from "axios"
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import RemoveIcon from '@material-ui/icons/Remove';
import AddIcon from '@material-ui/icons/Add';
import { v4 as uuidv4 } from 'uuid';




import { makeStyles } from '@material-ui/core/styles';
import { style } from "@mui/system";
import Datatable from "../../datatable";

const useStyles = makeStyles((theme) => ({
  root: {
    '& .MuiTextField-root': {
      margin: theme.spacing(1),
    },
  },
  button: {
    margin: theme.spacing(1, 4),
  }
}))

const ModelEvaluation = (props) => {
  const classes = useStyles()
  const [myModelData, setModelData] = useState([]);
  const [modelingState, setModelingState] = useState({
    rank1: false,
    rank2: false,
    rank3: false,
    rank4: false,
  })
  const [logLossState, setLogLossState] = useState([])
  const [rocScoreState, setRocScoreState] = useState([])
  const [acurracyScoreState, setAcurracyScoreState] = useState([])
  const[rocImage, setRocImage] = useState([])
  const[conImage, setConImage] = useState([])


  const [modelingVariable, setModelingVariable] = useState([])
  const [matrixData, setMatrixData] = useState([])
  const renderMatrix = (matrix, index) => {
    return (
      <tr key={index}>
        <td>{matrix.Percesion}</td>
        <td>{matrix.Recall}</td>
        <td>{matrix.F1Score}</td>
        <td>{matrix.Accuracy}</td>
      </tr>
    )
  }
  useEffect(async () => await axios.get("http://127.0.0.1:8000/api/model_res")
    .then((response) => {
      console.log(response.data.data);
    setModelData(response.data.data);
    console.log(response.data.summary);
    setModelingVariable(response.data.summary)
    })
    .catch((error) => {
      console.log(error)
    }),
    []);



  const [inputFields, setInputFields] = useState([
    { id: uuidv4(), selectVariable: '', evaluationValue: '', sampling: '', binaryOperation: ''},
  ]);

  const handleSubmit = (e) => {
    e.preventDefault();
    var rankedValue = document.getElementById("target-dropdown").value;
    console.log(rankedValue)
    var rankmetric1 = document.getElementById("rank1")
    var rankmetric2 = document.getElementById("rank2")

    if(rankmetric1.checked==true){
      var rankvalue = document.getElementById("rank1").value;
      console.log(rankvalue)
    }
    else if(rankmetric2.checked==true){
      var rankvalue = document.getElementById("rank2").value;
      console.log(rankvalue)
    }
    let modelevaluated = rankvalue;
    let modelRank = rankedValue;

    axios
      .post("http://127.0.0.1:8000/api/modelingevaluation", {
        "InputFileds": inputFields,modelevaluated,modelRank
      })
      .then((res) => {
        console.log("InputFields", inputFields,modelevaluated,modelRank);
        console.log(res);
     
      });
    
     
      axios.get("http://127.0.0.1:8000/api/image")

      .then((response) => {
     
        console.log(response.data.Image);
        setRocImage(response.data.Image.file_path_roc);
             setConImage(response.data.Image.file_path_con);
             setMatrixData(response.data.data);
     
     
     
     
      })



  
  }
  

  const handleChangeInput = (id, event) => {
    const newInputFields = inputFields.map(i => {
      if (id === i.id) {
        i[event.target.name] = event.target.value
      }
      return i;
    })

    setInputFields(newInputFields);
  }

  const handleAddFields = () => {
    setInputFields([...inputFields, { id: uuidv4(), selectVariable: '', evaluationValue: '', sampling: '', binaryOperation: '' }])
  }

  const handleRemoveFields = id => {
    const values = [...inputFields];
    values.splice(values.findIndex(value => value.id === id), 1);
    setInputFields(values);
  }

  const onChangeLogistic = () => {
    setModelingState(initialState => ({
      islogisticRegression: !initialState.islogisticRegression,
    }));
  }
  const onChangeXgboost = () => {
    setModelingState(initialState => ({
      isxgboost_classifier: !initialState.isxgboost_classifier,
    }));
  }
  const onChangeDecisionTree = () => {
    setModelingState(initialState => ({
      isdecision_tree: !initialState.isdecision_tree,
    }));
  }
  const onChangeCbclassifier = () => {
    setModelingState(initialState => ({
      iscb_classifier: !initialState.iscb_classifier,
    }));
  }

  const changeRank = (e) => {
    var rank = document.getElementById('target-dropdown')
    if (rank.value === "Log Loss") {
      console.log(myModelData[0].log_loss)
      setLogLossState(myModelData[0].log_loss)
      var displayaccuracy = document.getElementById("accuracyrank");
      displayaccuracy.style.display = "none";
      var dispalyroc = document.getElementById("rocrank")
      dispalyroc.style.display = "none";
      var displayloss = document.getElementById("loglossrank");
      displayloss.style.display = "block";
      var rankdiv = document.getElementById("rankingDiv")
      rankdiv.style.display = "block"


    }
    if (rank.value === "ROC AUC Score") {
      console.log(myModelData[0].roc_auc_score)
      setRocScoreState(myModelData[0].roc_auc_score)
      var displayloss = document.getElementById("loglossrank");
      displayloss.style.display = "none";
      var displayaccuracy = document.getElementById("accuracyrank");
      displayaccuracy.style.display = "none";
      var dispalyroc = document.getElementById("rocrank")
      dispalyroc.style.display = "block";
      var rankdiv = document.getElementById("rankingDiv")
      rankdiv.style.display = "block"

    }
    if (rank.value === "Accuracy Score") {
      console.log(myModelData[0].accuracy_score)
      setAcurracyScoreState(myModelData[0].accuracy_score)
      var displayloss = document.getElementById("loglossrank");
      displayloss.style.display = "none";
      var dispalyroc = document.getElementById("rocrank")
      dispalyroc.style.display = "none";
      var displayaccuracy = document.getElementById("accuracyrank");
      displayaccuracy.style.display = "block";
      var rankdiv = document.getElementById("rankingDiv")
      rankdiv.style.display = "block"


    }

  }
  const onChangeRank1 = () => {
    setModelingState(initialState => ({
      rank1: !initialState.rank1,
    }));
    var rankmetric1 = document.getElementById("rank1")
    if(rankmetric1.checked==true){
      var rankvalue = document.getElementById("rank1").value;
      console.log(rankvalue)
      setModelingState(rankvalue)
    }
  }
  const onChangeRank2 = () => {
    setModelingState(initialState => ({
      rank2: !initialState.rank2,
    }));
  }
  // const onChangeRank3 = () => {
    
  //   setModelingState(initialState => ({
  //     rank3: !initialState.rank3,
  //   }));
  // }
  // const onChangeRank4 = () => {
  //   setModelingState(initialState => ({
  //     rank4: !initialState.rank4,
  //   }));
  // }
  // const onChangeRank5 = () => {
  //   setModelingState(initialState => ({
  //     rank5: !initialState.rank5,
  //   }));
  // }
  // const onChangeRank6 = () => {
  //   setModelingState(initialState => ({
  //     rank6: !initialState.rank6,
  //   }));
  // }

  const handleFreeze = (e) => {
    e.preventDefault();
  
    var rankmetric1 = document.getElementById("rank1")
    var rankmetric2 = document.getElementById("rank2")

    if(rankmetric1.checked==true){
      var rankvalue = document.getElementById("rank1").value;
      console.log(rankvalue)
    }
    else if(rankmetric2.checked==true){
      var rankvalue = document.getElementById("rank2").value;
      console.log(rankvalue)
    }
    let modelevaluated = rankvalue;
    Object.freeze(inputFields,modelevaluated)
    console.log(inputFields)
    axios
      .post("http://127.0.0.1:8000/api/freeze", {
        "InputFileds": inputFields,modelevaluated
      })
      .then((res) => {
        console.log("InputFields", inputFields,modelevaluated);
      });
    
  }
  const handleReset = (e) =>{
window.location.reload();
  }


  return (
    <> <Grid id="modelEvaluation">
      <Grid class="sub-nav">
        <Grid class="subnav-modeling-content">
          <Typography>Model Ranking</Typography>

        </Grid>
      </Grid>

      <Grid class="target-selector-Evaluation" >
        <label for="target-dropdown">Select Ranking Metric: </label>
        <select id='target-dropdown' name="target-dropdown" onChange={changeRank}>
          <option value="select" id="select" selected></option>
          <option value="Log Loss" id="Log Loss" >Log Loss</option>
          <option value="ROC AUC Score" id="ROC AUC Score">ROC AUC Score</option>
          <option value="Accuracy Score" id="Accuracy Score">Accuracy Score</option>
        </select>
      </Grid>
      <div id="rankingDiv" style={{ display: "none" }}>
        <div id="loglossrank">
          <ul>
            {
              myModelData.filter(modeling => modeling.log_loss.rank1).map(filteredModeling => (

                <><li> <input type="checkbox" id="rank1" name="modelevaluated" onChange={onChangeRank1} value={filteredModeling.log_loss.rank1} checked={modelingState.rank1} />
                  {filteredModeling.log_loss.rank1} {filteredModeling.log_loss.rank1value}</li></>
              ))}
            {
              myModelData.filter(modeling => modeling.log_loss.rank2).map(filteredModeling => (
                <li> <input type="checkbox" id="rank2" onChange={onChangeRank2} name="modelevaluated" value={filteredModeling.log_loss.rank2} checked={modelingState.rank2} />{filteredModeling.log_loss.rank2} {filteredModeling.log_loss.rank2value}</li>
              ))}
          </ul>
        </div>
        <div id="accuracyrank">
             <ul>
            {
              myModelData.filter(modeling => modeling.accuracy_score.rank1).map(filteredModeling => (

                <><li> <input type="checkbox" id="rank1" name="modelevaluated" onChange={onChangeRank1} value={filteredModeling.accuracy_score.rank1} checked={modelingState.rank1} />
                  {filteredModeling.accuracy_score.rank1} {filteredModeling.accuracy_score.rank1value}</li></>
              ))}
            {
              myModelData.filter(modeling => modeling.accuracy_score.rank2).map(filteredModeling => (
                <li> <input type="checkbox" id="rank2" onChange={onChangeRank2} name="modelevaluated" value={filteredModeling.accuracy_score.rank2} checked={modelingState.rank2} />{filteredModeling.accuracy_score.rank2} {filteredModeling.accuracy_score.rank2value}</li>
              ))}
          </ul>
        </div>
        <div id="rocrank">
            <ul>
            {
              myModelData.filter(modeling => modeling.roc_auc_score.rank1).map(filteredModeling => (

                <><li> <input type="checkbox" id="rank1" name="modelevaluated" onChange={onChangeRank1} value={filteredModeling.roc_auc_score.rank1} checked={modelingState.rank1} />
                  {filteredModeling.roc_auc_score.rank1} {filteredModeling.roc_auc_score.rank1value}</li></>
              ))}
            {
              myModelData.filter(modeling => modeling.roc_auc_score.rank2).map(filteredModeling => (
                <li> <input type="checkbox" id="rank2" onChange={onChangeRank2} name="modelevaluated" value={filteredModeling.roc_auc_score.rank2} checked={modelingState.rank2} />{filteredModeling.roc_auc_score.rank2} {filteredModeling.roc_auc_score.rank2value}</li>
              ))}
          </ul>
        </div>
      </div>
      {inputFields.map((inputField, index) => (
        <div key={index}>
          <Grid class="targetVariation">
            <Grid class="target-selector-2-modelevaluation" >
              <label id="target-dropdown-2" className="selectVariableLabel" >Select Variable: </label>
              <select id='selectVariable' name="selectVariable" onChange={event => handleChangeInput(inputField.id, event)}>
                <option value="select variable">select variable</option>
                {modelingVariable.map(newOption => (<option value={newOption.userId} key={newOption} >{newOption.userId}</option>))}

              </select>
            </Grid>

            <Grid class="target-selector-2-1" >
              <label id="target-dropdown-2" className="selectVariableLabel">Enter Value: </label>
              <input
                name="evaluationValue"
                type="text"
                id="evaluationValue"
                placeholder="Enter value"
                onChange={event => handleChangeInput(inputField.id, event)}
              ></input>
            </Grid>



            <Grid class="taget-selector-3-modelevaluation">
              <label id='Sampling' className="selectVariableLabel">Select operator: </label>
              <select name="sampling" id='sampling' onChange={event => handleChangeInput(inputField.id, event)}>
                <option value="Select operator">Select operator</option>
                <option value="<">&lt;</option>
                <option value=">">	&gt;</option>
                <option value="=">=</option>
                <option value="<=">&lt;=</option>
                <option value=">=">&gt;=</option>
              </select>
            </Grid>
            <Grid class="target-selector-3-1" >
              <label for="target-dropdown-2" className="selectVariableLabel">Condition:</label>
              <select id='binaryOperation' name="binaryOperation" onChange={event => handleChangeInput(inputField.id, event)}>
                <option value="select operation">select operation</option>
                <option value="AND">AND</option>
                <option value="OR">OR</option>

              </select>
            </Grid>
            <IconButton disabled={inputFields.length === 1} onClick={() => handleRemoveFields(inputField.id)}>
              <RemoveIcon />
            </IconButton>
            <IconButton
              onClick={handleAddFields}
            >
              <AddIcon />
            </IconButton>

          </Grid>
        </div>
      ))}
      <Button
        className={classes.button}
        variant="contained"
        color="primary"
        type="submit"
        onClick={handleSubmit}
      >Send</Button>
      <Button
        className={classes.button}
        variant="contained"
        color="primary"
        type="submit"
        onClick={handleReset}
      >Reset</Button>

      <Grid>
        <Button variant="contained" className={classes.button} onClick={handleFreeze} color="secondary" type="submit" >Freeze Data</Button>
      </Grid>

      <Grid className="dataOutput">
        <Grid className="datSummary">
          <Typography className="dataSummaryText">Model Performance</Typography>
        </Grid>
        <Grid className="dataValue">
          <div className="dataOutput1">
            <h4>Roc Curve:-</h4>
            <img class="rocImage" src={rocImage} />

          </div>
          <div className="dataOutput2">
           
               <table>
                 <tr>
                   <td>Percesion</td>
                   {matrixData.map(pre =><td>{pre.precision}</td>)}
                   </tr>
                   <tr> <td>Recall</td>
                   {matrixData.map(rec =><td>{rec.recall}</td>)}
                  </tr>

               <tr> <td>F1Score</td>
               {matrixData.map(f1s =><td>{f1s.f1_score}</td>)}
               </tr>
                  <tr>
                   <td>Accuracy</td>
                   {matrixData.map(acc =><td>{acc.accuracy}</td>)}
                 </tr>
                 </table>
          
          </div>
          <div className="dataOutput3">
            <h4>Confusion Matrix:-</h4>
            <img class="conImage" src={conImage} />


          </div>
        </Grid>

      </Grid>
    </Grid>
    </>

  );
}


export default ModelEvaluation;