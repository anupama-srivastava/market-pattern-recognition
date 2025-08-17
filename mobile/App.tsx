import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider } from 'react-redux';
import { store } from './src/store/store';
import { ThemeProvider } from './src/contexts/ThemeContext';
import { AuthProvider } from './src/contexts/AuthContext';
import { MarketProvider } from './src/contexts/MarketContext';
import { NotificationProvider } from './src/contexts/NotificationContext';

// Screens
import HomeScreen from './src/screens/HomeScreen';
import PortfolioScreen from './src/screens/PortfolioScreen';
import AlertsScreen from './src/screens/AlertsScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import MarketScreen from './src/screens/MarketScreen';

// Components
import TabBar from './src/components/TabBar';
import { StatusBar } from 'expo-status-bar';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <Provider store={store}>
      <ThemeProvider>
        <AuthProvider>
          <MarketProvider>
            <NotificationProvider>
              <NavigationContainer>
                <StatusBar style="auto" />
                <Tab.Navigator
                  tabBar={(props) => <TabBar {...props} />}
                  screenOptions={{
                    headerShown: false,
                  }}
                >
                  <Tab.Screen name="Home" component={HomeScreen} />
                  <Tab.Screen name="Market" component={MarketScreen} />
                  <Tab.Screen name="Portfolio" component={PortfolioScreen} />
                  <Tab.Screen name="Alerts" component={AlertsScreen} />
                  <Tab.Screen name="Settings" component={SettingsScreen} />
                </Tab.Navigator>
              </NavigationContainer>
            </NotificationProvider>
          </MarketProvider>
        </AuthProvider>
      </ThemeProvider>
    </Provider>
  );
}
