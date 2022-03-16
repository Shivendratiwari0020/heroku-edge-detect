import React, { Component } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { IconButton } from "@material-ui/core";
import RemoveIcon from '@material-ui/icons/Remove';
import AddIcon from '@material-ui/icons/Add';
import Button from '@material-ui/core/Button';
import './RuleBased.css';
import { Grid } from '@material-ui/core';
import axios from "axios"




class RuleBased extends Component {

	constructor(props) {
		super(props);
		this.state = {
			rules: [
				{
					ruleRegion: [],
					ruleInputs: [{ id: uuidv4(), selectVariable: '', evaluationValue: '', selectOperator: '', selectOperation: '', selectedVariable: '' }]
				}
			],
			variableData: []
		};
	}

	addRule = (i) => {
		const rule = [...this.state.rules];
		rule[i].ruleInputs.push({ id: uuidv4(), selectVariable: '', evaluationValue: '', selectOperator: '', selectOperation: '', selectedVariable: '' })
		this.setState({ rule });
		console.log("All the Rulles: ", this.state.rules);

	}
	removeRule = (i, idx) => {
		const rule = [...this.state.rules];
		rule[i].ruleInputs.splice(idx, 1)
		this.setState({ rule });
		console.log(rule)

	}
	removeMainRule = (index) => {
		const rule = [...this.state.rules];
		rule.splice(index,1)
		this.setState({ rules: rule });

	}
	addNewRule = () => {
		const rules = [...this.state.rules];
		rules.push({
			ruleRegion: [],
			ruleInputs: [{ id: uuidv4(), selectVariable: '', evaluationValue: '', selectOperator: '', selectOperation: '', selectedVariable: '' }]
		});


		this.setState({ rules });
		console.log("All the Rulles: ", this.state.rules);
		console.log("new Rule: ", rules);
	};

	onChange = (id, e) => {

		const newRuleBaseField = this.state.rules.map(i => i.ruleInputs.map(data => {
			if (id === data.id) {
				data[e.target.name] = e.target.value
			}
			return data;
		}))

		this.setState(newRuleBaseField);
		console.log(this.state.rules)
	}




	componentDidMount() {
		fetch('http://127.0.0.1:8000/api/model_res')
			.then((response) => response.json())
			.then(variableList => {
				this.setState({ variableData: variableList.summary });
				console.log(this.state.variableData)
			});
	}








	submitData = (e) => {
		e.preventDefault();
		axios
			.post("http://127.0.0.1:8000/api/rulebased", {
				"Rules": this.state.rules
			})
			.then((res) => {
				console.log("Rules", this.state.rules);
			});


	}

	render() {
		return (


			<div className="rulebaseContent">

				<div className="forfeitbutton">
					<button className="forfeitClassifier">
						ForeFit Classifier

					</button>
				</div>
				<Grid>
					{this.state.rules.map((ruleInput, i) => (


						<div key={i} id="ruleVariation">
							<div className='ruleTitle'><h3>Rules</h3></div>

							{ruleInput.ruleInputs.map((ruledInput, idx) => (

								<Grid><div key={idx} id="rules1">
									<div id="ruleBased">


										<div className="target-selector-rulebased">
											<label className="target-dropdown-rule">Select Variable</label>
											<select id='selectVariable' name="selectVariable" onChange={(e) => this.onChange(ruledInput.id, e)}>
												<option value="0">Select Variable</option>
												{this.state.variableData.map(dataVariable => (<option value={dataVariable.userId} key={dataVariable.userId}>{dataVariable.userId}</option>))}

											</select>
										</div>

										<div className="target-selector-rulebased3">
											<label className="selectVariableLabel">Enter Value: </label>
											<input
												name="evaluationValue"
												type="text"
												id="evaluationValue"
												placeholder="Enter value"
												onChange={(e) => this.onChange(ruledInput.id, e)}



											></input>
										</div>

									</div>
									<div id="ruleBased">
										<div className="target-selector-rulebased1">
											<label className="target-dropdown-rulebased1">Select Operator</label>
											<select id='selectOperator' name="selectOperator" onChange={(e) => this.onChange(ruledInput.id, e)}>
												<option value="<">&lt;</option>
												<option value=">">	&gt;</option>
												<option value="=">=</option>
												<option value="<=">&lt;=</option>
												<option value=">=">&gt;=</option>
											</select>
										</div>
										<div className="target-selector-rulebased4">
											<label className="target-dropdown-rulebased4">Condition</label>
											<select id='selectOperation' name="selectOperation" onChange={(e) => this.onChange(ruledInput.id, e)}>
												<option value="select operation">select operation</option>
												<option value="AND">AND</option>
												<option value="OR">OR</option>
											</select>
										</div>
									</div>
									<div className="target-selector-rulebased2">
										<label className="target-dropdown-rulebased2">Targeted Value</label>
										<input
											name="selectedVariable"
											type="text"
											id="selectedVariable"
											placeholder="Enter value"
											onChange={(e) => this.onChange(ruledInput.id, e)}


										></input>
									</div>
									<div className="iconplus">
										<IconButton onClick={this.addRule.bind(this, i)}><AddIcon /> </IconButton>
										<IconButton disabled={ruleInput.ruleInputs.length === 1} onClick={this.removeRule.bind(this, i, idx)}>
											<RemoveIcon />
										</IconButton>
									</div>


								</div>


								</Grid>
							))}

						</div>


					))}
					<div className="addrule">

						<button onClick={this.addNewRule.bind(this)}>Add New Rule</button>
						&nbsp;
						<button onClick={this.removeMainRule.bind(this)}>Remove Rule</button>
					</div>



					<Button
						className='sendButton'

						variant="contained"
						color="primary"
						type="submit"
						onClick={this.submitData.bind(this)}

					>Send</Button></Grid>
			</div>

		);
	}
}

export default RuleBased;