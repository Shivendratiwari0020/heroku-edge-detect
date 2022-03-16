import React from "react";
import './homepage.css';
import bgImage from "../../assets/images/bg16.jpg";
import { useHistory } from "react-router"; 


const Home = () => {
  const history = useHistory();
  const handleClick = () =>{
history.push('/login')
  }
    return ( 
    <> 
    <div className="backgroundImage">
      <img src={bgImage} alt="Anomaly Image"></img>
    </div>
    <div className="edgeDetail">
        <div className="welcomeHeader"><h1>Welcome to Working EdgeDetect</h1></div>
        
   
        <div>Detecting Rare Events,
Anomalous Behaviors,
and Unusual Patterns</div>
<div className="homeLoginButton">
<button className="loginButton" size="large" variant ="outlined" onClick={handleClick} >Click here for Login/Signup</button>
</div>
        </div>
        
        </>
    );

}

export default Home;
