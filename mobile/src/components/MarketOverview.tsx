import React from 'react';
import { View, Text, StyleSheet, FlatList } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';

interface MarketOverviewProps {
  data: any;
}

const MarketOverview: React.FC<MarketOverviewProps> = ({ data }) => {
  const { theme } = useTheme();

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.card }]}>
      <Text style={[styles.title, { color: theme.colors.text }]}>Market Overview</Text>
      <View style={styles.metrics}>
        <View style={styles.metric}>
          <Text style={[styles.metricLabel, { color: theme.colors.text }]}>Total Market Cap</Text>
          <Text style={[styles.metricValue, { color: theme.colors.primary }]}>$2.5T</Text>
        </View>
        <View style={styles.metric}>
          <Text style={[styles.metricLabel, { color: theme.colors.text }]}>24h Volume</Text>
          <Text style={[styles.metricValue, { color: theme.colors.primary }]}>$150B</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    marginVertical: 8,
    borderRadius: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  metrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metric: {
    flex: 1,
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default MarketOverview;
