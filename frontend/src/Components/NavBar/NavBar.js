import React, {useState} from "react";
import "./NavBar.css";
import apexonicon from '../../assets/images/apexonicon.png'
import userAccount from '../../assets/images/userprofile.png'
import {NavLink} from 'react-router-dom';
import { IconButton } from "@material-ui/core";
import Logout from "../Logout/Logout";





const NavBar = () => {
  const [user, setUser] = useState({});
  return (
    <>
  <nav className="navBar">
  
     
        <NavLink className="header-logo" to="/" exact>
          <img src={apexonicon} alt="Apexon | EdgeDetect"></img>
        </NavLink>
      
 {user && 
      <div id="navbarSupportedContent">
        <ul className="navbar-nav ml-auto">
           

            <li className="nav-item">
              <NavLink activeStyle={{backgroundColor:"#add8e6"}} className="nav-link" to="/data" exact active>
                <i  className="data">
                </i>DATA
              </NavLink> 
            </li>

            <li className="nav-item">
              <NavLink activeStyle={{backgroundColor:"#add8e6"}} className="nav-link" to="/modeling" exact>
                <i className="modeling">
                </i>MODELING
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink activeStyle={{backgroundColor:"#add8e6"}} className="nav-link" to="/rulebased" exact>
                <i className="rulebased">
                </i>RULE BASED
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink activeStyle={{backgroundColor:"#add8e6"}} className="nav-link" to="/modelevaluation" exact>
                <i className="modelevaluation">
                </i>MODEL EVALUATION
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink activeStyle={{backgroundColor:"#add8e6"}} className="nav-link" to="/segmentclassifier" exact>
                <i className="segmentclassifier">
                </i>FINAL DETECTION
              </NavLink>
              </li>
              <li className="nav-item">
              {/* <NavLink activeStyle={{backgroundColor:"#add8e6"}} className="nav-link" to="/modelmonitoring" exact>
                <i className="modelmonitoring">
                </i>MODEL MONITORING
              </NavLink> */}
            </li>
            
        </ul>
        </div>
}
        {!user &&
        <NavLink className="loginAccount" to="/login" exact>
          <IconButton><img src={userAccount} alt="User" className="loginAccountImg"></img></IconButton>
        </NavLink>
}
{user &&
  <NavLink className="logout" to="/logout" exact>
          <IconButton><img src={userAccount} alt="User" className="loginAccountImg"></img></IconButton>
        </NavLink>
}
       
    </nav>
    </>
  );
};

export default NavBar;
