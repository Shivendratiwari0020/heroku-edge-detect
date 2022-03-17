import React, { useState, useEffect, Fragment, createRef } from 'react';
import "./data.css";
import bgImage from "../../../assets/images/bg16.jpg";
import axios, {post} from 'axios';


class  Data extends React.Component  {
   constructor(props){
     super(props);
      // this.hiddenFileInput = createRef();
      this.state = {
        selectedFile:'',
    }

    this.handleChange = this.handleChange.bind(this);
   }



  // useEffect(() => {
  //   if (localStorage.getItem('token') === null) {
  //     window.location.replace('http://localhost:3000/login');
  //   } else {
  //     fetch('http://127.0.0.1:8000/api/v1/users/auth/user/', {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //         Authorization: `Token ${localStorage.getItem('token')}`
  //       }
  //     })
  //       .then(res => res.json())
  //       .then(data => {
  //         setUserEmail(data.email);
  //         setLoading(false);
  //       });
  //   }z
  // }, []);

  // handleClick = (event) => {
  //   //this.hiddenFileInput.current.click();
  // };
  handleClick(){
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

  // handleChange = (e) => {
  //   let files = e.target.files;
  //   let reader= new FileReader();
  //   reader.readAsDataURL(files[0]);

  //   reader.onload = (e) => {
  //     console.log("data files", e.target.result);
  //     const url = "http://127.0.0.1:8000/api/upload";
  //     const formData = new FormData();
  //     formData.append('files',e.target.result)
  //     const config = {
  //       headers: {
  //           'content-type': 'multipart/form-data'
  //       }
  //   }
  //     return post(url,formData,config)
  //         .then(respone=> console.log("result",respone,formData)) 
  //   }


  //   console.log("data files", files)

  // };
   render(){
  return (
    <>
      <div className="backgroundImage">
        <img src={bgImage} alt="Anomaly Image"></img>
      </div>
      {/* {loading === false && ( */}
        <Fragment>
      <div className="edgeDetailData">
        <div className="welcomeHeader">
          <h1>EdgeDetect</h1>
        </div>

        <div className="dataHeadeing">
          Detecting Rare Events, Anomalous Behaviours, and Unusual Patterns.
          Click on the button below to upload the dataset.
        </div>
        <div className="homeLoginButton">
        <input
        type="file"
       // style={{ display: "none" }}
        ref={this.hiddenFileInput}
        onChange={(e)=>this.handleChange(e)}
        accept=".csv"
       />

           <button type="submit"  className="uploadButton" size="large" variant="outlined" onClick={()=>this.handleClick()}>
            Click here to Upload the data
          </button> 
        
        </div>
      </div>
      </Fragment>
      {/* )} */}
    </>
  );
      }
};

export default Data;
