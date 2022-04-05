import React, { useState, useEffect } from 'react';
import apexonicon from "../../assets/images/apexonicon.png";
import bgImage from "../../assets/images/bg16.jpg";
import './Login.css'
import { useHistory } from "react-router";


const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (localStorage.getItem('token') !== null) {
      window.location.replace('http://localhost:3000/data');
    } else {
      setLoading(false);
    }
  }, []);
  const history = useHistory();

  const onSubmit = e => {
    e.preventDefault();
    history.push("/homepage")

    // const user = {
    //   email: email,
    //   password: password
    // };

  //   fetch('http://127.0.0.1:8000/api/v1/users/auth/login/', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json'
  //     },
  //     body: JSON.stringify(user)
  //   })
  //     .then(res => res.json())
  //     .then(data => {
  //       if (data.key) {
  //         localStorage.clear();
  //         localStorage.setItem('token', data.key);
  //         window.location.replace('http://localhost:3000/data');
  //       } else {
  //         setEmail('');
  //         setPassword('');
  //         localStorage.clear();
  //         setErrors(true);
  //       }
  //     });
  };

  return (
    <div className="loginContainer">
      <div className="loginBgImage">
        <img src={bgImage} alt="Anomaly Image"></img>
      </div>
      {loading === false}
      {errors === true && <h2>Cannot log in with provided credentials</h2>}
      {loading === false && (
        <form onSubmit={onSubmit} className="loginForm">
          <div className="loginIcon">
          <img src={apexonicon}></img>
          </div>
          <label htmlFor='email'>Email address:</label> <br />
          <input
            name='email'
            type='email'
            value={email}
            required
            onChange={e => setEmail(e.target.value)}
          />{' '}
          <br />
          <label htmlFor='password'>Password:</label> <br />
          <input
            name='password'
            type='password'
            value={password}
            required
            onChange={e => setPassword(e.target.value)}
          />{' '}
          <br />
          <input type='submit' value='Login' />
          <a href="/signup">Signup</a>
        </form>
      )}
    </div>
  );
};

export default Login;