import * as React from 'react';
import { createAppContainer } from 'react-navigation';
import { createStackNavigator } from 'react-navigation-stack';
import CameraScreen from "./Components/CameraScreen";
import HomeScreen from "./Components/HomeScreen";
import DetailsScreen from "./Components/DetailsScreen";

const AppNavigation = createStackNavigator(
    {
      Home: {
        screen: HomeScreen,
        navigationOptions: {
          header: null,
        }
      },
      Details: {
        screen: DetailsScreen,
        navigationOptions: {
          header: null,
        }
      },
      Camera: {
        screen: CameraScreen,
        navigationOptions: {
          header: null,
        }
      },
    },
    {
      initialRouteName: 'Camera',
    }
);

const AppContainer = createAppContainer(AppNavigation);

export default function App() {
    return <AppContainer />;
}