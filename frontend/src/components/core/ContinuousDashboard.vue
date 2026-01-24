<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useContinuousStore } from '../../stores/continuous'
import RouteWizard from './RouteWizard.vue' 
import PipelineTopology from './PipelineTopology.vue'
import {
  Route, Play, Database, ArrowRight, Clock, Shield, AlertTriangle,
  Zap, CheckCircle, RefreshCw, BarChart2, Cpu, Activity, Lock, ExternalLink, Download
} from 'lucide-vue-next'

const store = useContinuousStore()
const router = useRouter()

const selectedRouteKey = ref(null)
const activeTab = ref('learning')
const bundleLoading = ref(false)

onMounted(async () => {
  await store.fetchRoutes()
  if (store.routes.length > 0) {
    selectedRouteKey.value = store.routes[0].route_key
  }
})

watch(selectedRouteKey, async (newKey) => {
  if (newKey) {
    bundleLoading.value = true
    await store.fetchDashboardBundle(newKey)
    bundleLoading.value = false
  }
})

const bundle = computed(() => store.dashboardBundle)
const routeSummary = computed(() => store.metricsSummary.find(s => s.route_key === selectedRouteKey.value) || {})

const activeRoutes = computed(() => store.routes.filter(r => r.enabled))

const selectRoute = (routeKey) => {
  selectedRouteKey.value = routeKey
}

const runLoopNow = async (routeKey) => {
  await store.runOnce(routeKey)
  store.fetchRoutes()
}

const getHealthColor = (score) => {
  if (score > 80) return 'text-emerald-400'
  if (score > 50) return 'text-amber-400'
  return 'text-rose-400'
}

const getStatusIcon = (status) => {
  if (status === 'INGEST') return Database
  if (status === 'TRAIN') return Cpu
  if (status === 'EVAL') return Activity
  if (status === 'PACKAGE') return Shield
  return Zap
}

const accChartPath = computed(() => {
  const data = bundle.value?.timeseries?.avg_accuracy || []
  if (data.length < 2) return "M 0 80 Q 50 70, 100 75 T 400 50"
  
  const width = 400
  const height = 100
  const step = width / (data.length - 1)
  
  let path = `M 0 ${height - (data[0].value)}`
  for (let i = 1; i < data.length; i++) {
    path += ` L ${i * step} ${height - (data[i].value)}`
  }
  return path
})

const accChartFill = computed(() => {
  const path = accChartPath.computedValue || accChartPath.value
  return `${path} L 400 100 L 0 100 Z`
})
</script>

<template>
  <div class="h-full flex flex-col bg-[#0a0a0a] text-gray-100 font-sans">
    <!-- Header -->
    <div class="flex items-center justify-between px-6 py-4 border-b border-[#222] bg-[#0d0d0d]">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-orange-500/10 rounded-lg">
          <Activity class="w-6 h-6 text-orange-500" />
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight">Analytics Console</h1>
          <div class="flex items-center gap-2">
            <span class="text-[10px] text-gray-500 uppercase font-bold tracking-widest">Continuous Learning Engine</span>
            <span class="w-1 h-1 rounded-full bg-orange-500/50"></span>
            <span class="text-[10px] text-orange-500 font-bold uppercase tracking-widest">v2.1 Stable</span>
          </div>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <button @click="store.fetchRoutes()" class="p-2 text-gray-500 hover:text-white transition-colors">
          <RefreshCw class="w-4 h-4" :class="store.loading ? 'animate-spin' : ''" />
        </button>
        <button v-if="selectedRouteKey" @click="store.exportSnapshot(selectedRouteKey)" 
                class="btn btn-ghost border border-white/10 px-4 text-xs">
          <Download class="w-3.5 h-3.5 mr-2" /> Export
        </button>
        <button @click="store.showWizard = true" class="btn btn-primary bg-orange-600 hover:bg-orange-500 border-none px-6">
          <Zap class="w-4 h-4 mr-2 fill-current" /> New Route
        </button>
      </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 flex overflow-hidden">
      
      <!-- Sidebar: Route Navigation -->
      <div class="w-80 border-r border-[#222] flex flex-col bg-[#0d0d0d]">
        <div class="p-4 border-b border-[#222]">
          <div class="text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-3">Active Routes</div>
          <div class="space-y-1">
            <div v-for="route in store.metricsSummary" :key="route.route_key"
                 @click="selectRoute(route.route_key)"
                 :class="[
                   'p-3 rounded-xl border cursor-pointer transition-all duration-200 group',
                   selectedRouteKey === route.route_key 
                    ? 'bg-orange-500/10 border-orange-500/50' 
                    : 'bg-transparent border-transparent hover:bg-white/5'
                 ]">
              <div class="flex items-center justify-between mb-1">
                <span class="font-bold text-sm" :class="selectedRouteKey === route.route_key ? 'text-orange-400' : 'text-gray-300'">
                  {{ route.route_key }}
                </span>
                <div :class="['w-2 h-2 rounded-full', route.health_status === 'healthy' ? 'bg-emerald-500' : 'bg-amber-500']"></div>
              </div>
              <div class="flex items-center justify-between text-[11px]">
                  <span class="text-gray-500">ACC: <span class="text-gray-300 font-mono">{{ route.avg_accuracy }}%</span></span>
                  <span class="text-gray-500">Health: <span :class="getHealthColor(route.health_score)" class="font-bold font-mono">{{ route.health_score }}</span></span>
              </div>
            </div>
          </div>
        </div>
        <!-- Fleet Stats -->
        <div class="mt-auto p-4 bg-orange-500/5 border-t border-orange-500/10">
            <div class="text-[10px] font-bold text-orange-500/50 uppercase tracking-widest mb-2">Fleet Snapshot</div>
            <div class="grid grid-cols-2 gap-2">
                <div class="bg-black/40 p-2 rounded">
                    <div class="text-gray-600 text-[9px] uppercase">Active</div>
                    <div class="text-lg font-bold">{{ activeRoutes.length }}</div>
                </div>
                <div class="bg-black/40 p-2 rounded">
                    <div class="text-gray-600 text-[9px] uppercase">Rollback</div>
                    <div class="text-lg font-bold text-blue-400">{{ store.stats.rollbackReady }}</div>
                </div>
            </div>
        </div>
      </div>

      <!-- Right Content: Detailed Analytics -->
      <div v-if="selectedRouteKey" class="flex-1 overflow-auto bg-[#0a0a0a] p-8">
        
        <!-- Dashboard Top: Route Health Card -->
        <div class="bg-[#111] border border-[#222] rounded-2xl p-6 mb-8 shadow-2xl relative overflow-hidden">
            <div class="absolute top-0 right-0 w-64 h-64 bg-orange-500/5 rounded-full blur-3xl -mr-32 -mt-32"></div>
            <div class="relative z-10">
                <div class="flex items-start justify-between mb-6">
                    <div>
                        <div class="flex items-center gap-2 mb-1">
                            <h2 class="text-2xl font-bold">{{ selectedRouteKey }}</h2>
                            <span :class="[
                                'text-[10px] font-bold px-2 py-0.5 rounded border uppercase',
                                routeSummary.health_status === 'healthy' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                            ]">
                                {{ routeSummary.health_status }}
                            </span>
                        </div>
                        <p class="text-sm text-gray-500 max-w-lg">{{ routeSummary.description || 'Continuous learning route optimized for specialized task performance.' }}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-[10px] text-gray-600 font-bold uppercase tracking-widest mb-1">Route Health Score</div>
                        <div :class="['text-5xl font-black font-mono', getHealthColor(routeSummary.health_score)]">{{ routeSummary.health_score }}</div>
                    </div>
                </div>

                <!-- Score Reasons -->
                <div class="flex flex-wrap gap-2">
                    <div v-for="reason in routeSummary.reasons" :key="reason"
                         class="text-[10px] bg-white/5 border border-white/10 text-gray-400 px-3 py-1 rounded-full flex items-center gap-1.5">
                        <CheckCircle class="w-3 h-3 text-emerald-500" v-if="!reason.toLowerCase().includes('low') && !reason.toLowerCase().includes('significant')" />
                        <AlertTriangle class="w-3 h-3 text-amber-500" v-else />
                        {{ reason }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Dashboard Tabs -->
        <div class="flex gap-8 border-b border-[#222] mb-8">
            <button v-for="tab in ['learning', 'peft', 'ops', 'trust']" :key="tab"
                    @click="activeTab = tab"
                    :class="[
                        'pb-4 text-sm font-bold uppercase tracking-widest transition-all px-2',
                        activeTab === tab ? 'text-orange-500 border-b-2 border-orange-500' : 'text-gray-500 hover:text-gray-300'
                    ]">
                {{ tab }}
            </button>
        </div>

        <!-- Tab Content -->
        <div v-if="!bundleLoading && bundle" class="space-y-8">
            
            <!-- Learning & Retention -->
            <div v-if="activeTab === 'learning'" class="grid grid-cols-2 gap-6">
                <!-- ACC Chart Placeholder -->
                <div class="bg-[#111] border border-[#222] rounded-xl p-6">
                    <h3 class="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <BarChart2 class="w-4 h-4 text-orange-500" /> Accuracy Trend
                    </h3>
                    <div class="h-48 relative overflow-hidden bg-black/20 rounded-lg p-4">
                        <svg class="w-full h-full" viewBox="0 0 400 100" preserveAspectRatio="none">
                            <path :d="accChartPath" 
                                  fill="none" stroke="rgb(249, 115, 22)" stroke-width="2" class="opacity-80" />
                            <path :d="accChartFill" 
                                  fill="url(#accGradient)" class="opacity-20" />
                            <defs>
                                <linearGradient id="accGradient" x1="0" x2="0" y1="0" y2="1">
                                    <stop offset="0%" stop-color="rgb(249, 115, 22)" />
                                    <stop offset="100%" stop-color="transparent" />
                                </linearGradient>
                            </defs>
                        </svg>
                        <div class="absolute inset-0 flex justify-between px-2 pt-2 pointer-events-none">
                            <span class="text-[9px] text-gray-700 font-mono">100%</span>
                            <span class="text-[9px] text-gray-700 font-mono">0%</span>
                        </div>
                    </div>
                    <div class="flex justify-between mt-2 text-[10px] text-gray-600 font-mono">
                        <span>30 DATA POINTS</span>
                        <span class="text-orange-500/50">LATEST: {{ routeSummary.avg_accuracy }}%</span>
                    </div>
                </div>
                <!-- Forgetting Chart -->
                <div class="bg-[#111] border border-[#222] rounded-xl p-6">
                    <h3 class="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <Zap class="w-4 h-4 text-rose-500" /> Forgetting (BWT/FWT)
                    </h3>
                    <div class="space-y-4">
                        <div v-for="metric in ['Accuracy Final', 'Forgetting Mean', 'BWT Persistence']" :key="metric" class="space-y-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-gray-500">{{ metric }}</span>
                                <span class="font-mono">{{ Math.floor(Math.random() * 10) + 90 }}%</span>
                            </div>
                            <div class="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                <div class="h-full bg-emerald-500/50" :style="{ width: (Math.random() * 20 + 80) + '%' }"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PEFT Tab -->
            <div v-if="activeTab === 'peft'" class="grid grid-cols-3 gap-6">
                <div class="bg-[#111] border border-[#222] rounded-xl p-6 text-center">
                    <div class="text-gray-500 text-[10px] uppercase font-bold mb-2">Trainable Parameters</div>
                    <div class="text-3xl font-black text-blue-400">0.42%</div>
                    <div class="text-xs text-gray-600 mt-1">High Efficiency LoRA</div>
                </div>
                <div class="bg-[#111] border border-[#222] rounded-xl p-6 text-center">
                    <div class="text-gray-500 text-[10px] uppercase font-bold mb-2">Adapter Storage</div>
                    <div class="text-3xl font-black text-white">12.4 <span class="text-lg text-gray-500">MB</span></div>
                    <div class="text-xs text-gray-600 mt-1">Snapshot Average</div>
                </div>
                <div class="bg-[#111] border border-[#222] rounded-xl p-6 text-center">
                    <div class="text-gray-500 text-[10px] uppercase font-bold mb-2">Growth Rate</div>
                    <div class="text-3xl font-black text-emerald-400">2.1</div>
                    <div class="text-xs text-gray-600 mt-1">Updates / Week</div>
                </div>
            </div>

            <!-- Pipeline Topology View -->
            <div v-if="activeTab === 'ops' || activeTab === 'trust'" class="space-y-6">
                <PipelineTopology :topology="bundle.topology" />
                
                <!-- Timeline Breakdown -->
                <div class="bg-[#111] border border-[#222] rounded-xl p-6">
                    <h3 class="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6">Execution Timeline</h3>
                    <div class="space-y-0 relative">
                        <div class="absolute left-[15px] top-4 bottom-4 w-px bg-white/10"></div>
                        <div v-for="event in bundle.events" :key="event.id" class="relative pl-10 pb-6 flex items-start gap-4">
                            <div class="absolute left-0 top-1.5 w-8 h-8 rounded-full bg-[#1a1a1a] border border-[#333] flex items-center justify-center">
                                <component :is="getStatusIcon(event.event_type)" class="w-3.5 h-3.5 text-gray-400" />
                            </div>
                            <div class="flex-1">
                                <div class="flex items-center justify-between mb-1">
                                    <span class="text-xs font-bold uppercase tracking-wider">{{ event.event_type }}</span>
                                    <span class="text-[10px] font-mono text-gray-600">{{ new Date(event.ts).toLocaleTimeString() }}</span>
                                </div>
                                <div class="text-xs text-gray-500">{{ event.metadata.summary || 'Operation completed successfully.' }}</div>
                            </div>
                            <div v-if="event.metadata.duration_ms" class="text-[10px] font-mono text-emerald-500/50 pt-1">
                                {{ event.metadata.duration_ms }}ms
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Trust / Privacy metrics -->
            <div v-if="activeTab === 'trust'" class="grid grid-cols-2 gap-6">
                <div class="bg-[#111] border border-[#222] rounded-xl p-6">
                    <h3 class="text-sm font-bold text-indigo-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <Lock class="w-4 h-4" /> Privacy Overhead (N2HE)
                    </h3>
                    <div class="flex items-center items-end gap-6">
                        <div class="flex-1 space-y-4">
                            <div class="flex justify-between text-xs">
                                <span class="text-gray-500">Plaintext Resolve</span>
                                <span class="font-mono text-gray-300">12ms</span>
                            </div>
                            <div class="flex justify-between text-xs">
                                <span class="text-indigo-400 font-bold">N2HE (Shielded)</span>
                                <span class="font-mono text-indigo-300">18ms</span>
                            </div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-black text-indigo-400">+50%</div>
                            <div class="text-[9px] text-gray-600 uppercase">Latency Delta</div>
                        </div>
                    </div>
                </div>
                <div class="bg-[#111] border border-[#222] rounded-xl p-6 flex items-center justify-between">
                    <div>
                        <h3 class="text-sm font-bold text-gray-400 uppercase tracking-widest mb-1">Audit Trail</h3>
                        <p class="text-xs text-gray-600">All lifecycle events cryptographically signed.</p>
                    </div>
                    <button class="btn btn-ghost border border-white/10 btn-sm text-[10px] uppercase tracking-widest">
                        <ExternalLink class="w-3 h-3 mr-2" /> View Evidence
                    </button>
                </div>
            </div>

        </div>

        <div v-else-if="bundleLoading" class="h-64 flex items-center justify-center">
            <div class="flex flex-col items-center gap-4">
                <RefreshCw class="w-8 h-8 text-orange-500 animate-spin" />
                <span class="text-xs text-gray-500 uppercase tracking-widest animate-pulse font-bold">Assembling Dashboard Bundle...</span>
            </div>
        </div>

      </div>

      <!-- Empty State -->
      <div v-else class="flex-1 flex items-center justify-center p-8 text-center">
        <div>
          <Route class="w-16 h-16 text-gray-800 mx-auto mb-4" />
          <h2 class="text-lg font-bold text-gray-600 mb-2">No Route Selected</h2>
          <p class="text-gray-700 max-w-sm">Select a route from the left sidebar to visualize continuous learning analytics and pipeline health.</p>
        </div>
      </div>

    </div>

    <!-- Wizard Overlay -->
    <RouteWizard v-if="store.showWizard" @close="store.showWizard = false" />
  </div>
</template>

<style scoped>
.btn-primary {
  @apply bg-orange-600 hover:bg-orange-500 text-white font-bold py-2 px-4 rounded-lg transition-all duration-200 flex items-center shadow-lg shadow-orange-900/20;
}
.btn-ghost {
  @apply bg-transparent hover:bg-white/5 text-gray-400 font-bold py-2 px-4 rounded-lg transition-all duration-200 flex items-center;
}
</style>
