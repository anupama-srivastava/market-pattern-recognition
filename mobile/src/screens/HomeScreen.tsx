import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
} from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { useMarketData } from '../hooks/useMarketData';
import MarketOverview from '../components/MarketOverview';
import PatternAlerts from '../components/PatternAlerts';
import PortfolioSummary from '../components/PortfolioSummary';
import QuickActions from '../components/QuickActions';

const HomeScreen: React.FC = () => {
  const { theme } = useTheme();
  const { marketData, loading, refresh } = useMarketData();

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: theme.colors.background }]}
      refreshControl={
        <RefreshControl refreshing={loading} onRefresh={refresh} />
      }
    >
      <View style={styles.header}>
        <Text style={[styles.title, { color: theme.colors.text }]}>
          Market Overview
        </Text>
      </View>

      <MarketOverview data={marketData} />
      <PatternAlerts alerts={marketData.alerts} />
      <PortfolioSummary />
      <QuickActions />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: 20,
    paddingTop: 60,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
  },
});

export default HomeScreen;
