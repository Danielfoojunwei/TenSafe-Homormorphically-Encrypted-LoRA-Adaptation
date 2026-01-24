<script setup>
import { Activity, Database, Key, ShieldCheck } from 'lucide-vue-next'
import PipelineGraph from './dashboard/PipelineGraph.vue'
import PerformanceDissect from './analytics/PerformanceDissect.vue'

const stats = [
  { label: 'System Health', value: '99.9%', icon: Activity, color: 'text-green-500' },
  { label: 'Active Fleets', value: '12', icon: Database, color: 'text-blue-500' },
  { label: 'Keys Rotated', value: '24h', icon: Key, color: 'text-orange-500' },
  { label: 'Compliance', value: 'Level 4', icon: ShieldCheck, color: 'text-purple-500' },
]
</script>

<template>
  <div class="space-y-6">
    <header class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold">System Dashboard</h2>
        <span class="text-xs text-gray-500">Real-time Telemetry & Pipeline Status</span>
      </div>
    </header>

    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div v-for="stat in stats" :key="stat.label" class="bg-[#161b22] border border-[#30363d] p-4 rounded-lg flex items-center gap-4">
        <div class="p-2 rounded-md bg-opacity-10" :class="stat.color.replace('text-', 'bg-')">
          <component :is="stat.icon" class="w-5 h-5" :class="stat.color" />
        </div>
        <div>
          <div class="text-lg font-bold">{{ stat.value }}</div>
          <div class="text-xs text-gray-400">{{ stat.label }}</div>
        </div>
      </div>
    </div>

    <!-- Pipeline Visualizer -->
    <PipelineGraph />

    <!-- Mission Control Analytics -->
    <PerformanceDissect />
  </div>
</template>
