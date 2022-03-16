import React, { useState, useEffect } from 'react';
import { Grid } from '@material-ui/core';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import RemoveIcon from '@material-ui/icons/Remove';
import AddIcon from '@material-ui/icons/Add';
import './segmentClassifier.css'
import { makeStyles } from '@material-ui/core/styles';
import { v4 as uuidv4 } from 'uuid';
import axios from "axios";
import RuleBased from '../RuleBased/RuleBased';
import Datatable from '../../datatable';




const useStyles = makeStyles((theme) => ({
    root: {
        '& .MuiTextField-root': {
            margin: theme.spacing(1),
        },
    },
    button: {
        margin: theme.spacing(2, 6),
    },

}))


const SegmentClassifier = () => {
    const classes = useStyles()

    const [myData, setData] = useState([]);
    const [dataModels, setDataModels] = useState([])

    const [segmentFields, setSegmentFields] = useState([{
        id: uuidv4(), segmentRule: '', segmentModel: ''

    }])
    const [baseModels, setBaseModels] = useState([{
        baseModel: '',
    }])

    const [ruleData, setRuleData] = useState([])

    useEffect(() => axios.get("http://127.0.0.1:8000/api/sendfreezedata")
    .then((response) => {
      console.log(response);
      setRuleData(response.data.InputFileds)
      


    })
    .catch((error) => {
      console.log(error)
    }),
    []);

    useEffect(async () => await axios.get("http://127.0.0.1:8000/api/ruledata")
        .then((response) => {
            console.log(response)
            console.log(response.data.rules.Rules);
            // console.log(response.data.rules);
            // const result = response.data.rules.Rules.filter((thing, index, self) =>
            //     index === self.findIndex((t) => (
            //         t.place === thing.place && t.name === thing.name
            //     ))
            // )
            // console.log(result)
            // setData(result);
            // console.log(result)
            // setDataModels(response.data.rules.models);
            setData(response.data.rules.Rules)

        })
        .catch((error) => {
            console.log(error)
        }),
        []);

    


    const handleBaseModelChange = (id, event) => {
        const newBaseModel = baseModels.map(i => {
            if (id === id) {
                i[event.target.name] = event.target.value
            }
            return i;
        })
        setBaseModels(newBaseModel)


    }
    const handleChange = (id, event) => {
        const newInputFields = segmentFields.map(i => {
            if (id === i.id) {
                i[event.target.name] = event.target.value
            }
            return i;
        })

        setSegmentFields(newInputFields);
    }


    const addRule = () => {
        setSegmentFields([...segmentFields, {
            id: uuidv4(), segmentRule: '', segmentModel: ''

        }])
    }
    const removeRule = (index) => {
        const values = [...segmentFields];
        values.splice(index, 1);
        setSegmentFields(values);
    }
    const submitRule = (e) => {
        e.preventDefault();

        if (baseModels.baseModel == "none") {
            console.log("Segment Rules:", segmentFields)
        }
        else {

            console.log("Segment Rules:", segmentFields, baseModels)
        }
        e.preventDefault();
        console.log(segmentFields, baseModels)
        axios
            .post("http://127.0.0.1:8000/api/modelevaluation", segmentFields, baseModels)
            .then((res) => {
                console.log("SegmentFields:", segmentFields, "BaseModels:", baseModels);
            });

    }

    //Download File Api
    const donwloadResult = () => {
        fetch('http://localhost:3000/assets/files/NiharSawant[1_0]', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/pdf',
            },
        })
            .then((response) => response.blob())
            .then((blob) => {
                // Create blob link to download
                const url = window.URL.createObjectURL(
                    new Blob([blob]),
                );
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute(
                    'download',
                    `FileName.py`,
                );

                // Append to html link element page
                document.body.appendChild(link);

                // Start download
                link.click();

                // Clean up and remove the link
                link.parentNode.removeChild(link);
            });
    }

    return (
        <div>
            <Grid className='segmentClassifier'>
                <Grid className='headerSegment'>
                    <h1>Segment Classifier</h1>
                </Grid>
                <Grid id="segment">
                    
                    <Datatable className="segmentRules" data={ruleData} />

                </Grid>

                <Grid  id="ruleSection">
                    <h1>Rules Section</h1>
                    <Datatable className="rules" data={myData} />

                </Grid>




































                <Grid id="finalResult">
                    <Grid className='finalresult' >
                        <label className='resultLabel'>
                            <h3>Get Your Evaluated Result</h3>
                        </label>

                        <Button className={classes.button}
                            variant="contained"
                            color="primary"
                            type="submit" className='downloadButton' onClick={donwloadResult}>Download</Button>

                    </Grid>
                </Grid>
            </Grid>

        </div>
    );
};

export default SegmentClassifier;