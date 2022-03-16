import {
  BrowserRouter as Router,
  Route,
  Redirect,
  Switch,
} from "react-router-dom";
import Data from "./Components/Dashboard/Data/Data";
import Home from "./Components/HomePage/homepage";
import Login from "./Components/Login/Login";
import ModelEvaluation from "./Components/Dashboard/ModelEvaluation/ModelEvaluation";
import Modeling from "./Components/Dashboard/Modeling/Modeling";
import NavBar from "./Components/NavBar/NavBar";
import Signup from "./Components/Signup/Signup";
import RuleBased from "./Components/Dashboard/RuleBased/RuleBased";
import Logout from "./Components/Logout/Logout";
import SegmentClassifier from "./Components/Dashboard/SegmentClassifier/SegmentClassifier"
const App = () => {
  return (
    <>
      <Router>
        <NavBar />
        <main>
          <Switch>
            <Route path="/" exact component={Home} />
            <Route path="/data" exact component={Data} />
            <Route path="/login" exact component={Login} />
            <Route path="/signup" exact component={Signup} />
            <Route path="/logout" exact component={Logout} />
            <Route path="/modelevaluation" exact component={ModelEvaluation} />
            <Route path="/modeling" exact component={Modeling} />
            <Route path="/rulebased" exact component={RuleBased} />
            <Route path="/modelmonitoring" exact component={Modeling} />
            <Route path="/segmentclassifier" exact component={SegmentClassifier} />

            <Redirect to="/" />
          </Switch>
        </main>
      </Router>
    </>
  );
};

export default App;
