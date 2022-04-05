import React, { Component } from "react";
import './homepage.css';
import bgImage from "../../assets/images/bg16.jpg";
import { useHistory } from "react-router";
import axios, { post } from 'axios';


// login button click handler
class Home extends Component  {
  constructor(props) {
    super(props);
    // this.hiddenFileInput = createRef();
    this.state = {
      selectedFile: '',
    }

    this.handleChange = this.handleChange.bind(this);
  }
  handleClick() {
    const data = new FormData()
    data.append('file', this.state.selectedFile)
    console.warn(this.state.selectedFile);
    let url = "http://127.0.0.1:8000/api/upload";

    axios.post(url, data, { // receive two parameter endpoint url ,form data 
    })
      .then(res => { // then print response status
        console.warn(res);
      })

  }
  handleChange(event) {
    this.setState({
      selectedFile: event.target.files[0],
    })
  }

  render() {
  return (
    <>
    {/* backgroundImage */}
      <div className="backgroundImage">
        <img src={bgImage} alt="Anomaly Image"></img>
      </div>
      {/* header */}
      <div className="edgeDetail">
        <div className="welcomeHeader"><h1>Welcome to EdgeDetect</h1></div>


        <div>Detecting Rare Events,
          Anomalous Behaviors,
          and Unusual Patterns</div>

          {/* Login Button Home */}
          <div className="UploadButton">
              <input
                type="file"

                ref={this.hiddenFileInput}
                onChange={(e) => this.handleChange(e)}
                accept=".csv"
              />

              <button type="submit" className="uploadButton" size="large" variant="outlined" onClick={() => this.handleClick()}>
                Click here to Upload the data
              </button>

            </div>
            </div>

    </>
  );
  }

};

export default Home;