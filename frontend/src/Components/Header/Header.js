import React from "react";
import NavBar from "../NavBar/NavBar";
import "./Header.css"
import apexonicon from '../../assets/images/apexonicon.png'
import userAccount from '../../assets/images/userprofile.png'
import Link from '@material-ui/core/Link';


const Header = () => {
    return ( 
    <> 
{/* <section className="header">
<section className="header-logo">
  <Link href="/Home"><img src={apexonicon} alt="EdgeDetect" className="EdgeDetect" /></Link> 
    </section>
    <section className="navBar">
    <NavBar />
    </section>
    <section className="userAccount">
   <Link href="/SignInOutContainer"> <img src={userAccount} alt="UserAccount" className="UserAccount" /> </Link>
    </section>
    </section>         */}
        </>
    
    );
  };
  
  export default Header;