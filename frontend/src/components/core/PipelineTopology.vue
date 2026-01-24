<script setup>
import { computed } from 'vue'
import { 
  Database, Cpu, Shield, Zap, Box, Activity, CheckCircle, AlertCircle
} from 'lucide-vue-next'

const props = defineProps({
  topology: {
    type: Object,
    default: () => ({ nodes: [], edges: [] })
  }
})

const getIcon = (category) => {
  switch (category) {
    case 'data_source': return Database
    case 'training_executor': return Cpu
    case 'model_registry': return Box
    case 'serving_exporter': return Zap
    case 'trust_privacy': return Shield
    default: return Activity
  }
}

const getCategoryColor = (category) => {
  switch (category) {
    case 'data_source': return 'text-blue-400 border-blue-500/30 bg-blue-500/5'
    case 'training_executor': return 'text-orange-400 border-orange-500/30 bg-orange-500/5'
    case 'model_registry': return 'text-purple-400 border-purple-500/30 bg-purple-500/5'
    case 'serving_exporter': return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/5'
    default: return 'text-gray-400 border-gray-500/30 bg-gray-500/5'
  }
}

// Simple horizontal layout calculation for SVG
const layoutNodes = computed(() => {
    const nodes = props.topology.nodes || []
    if (nodes.length === 0) return []
    
    // Group by category to determine X position
    const cats = ['data_source', 'training_executor', 'model_registry', 'serving_exporter']
    const spacingX = 220
    const spacingY = 100
    
    return nodes.map((node, i) => {
        const catIdx = cats.indexOf(node.category)
        const x = (catIdx !== -1 ? catIdx : 0) * spacingX + 50
        const y = 80 // Minimalist single row for now, or stagger
        return { ...node, x, y }
    })
})

const edges = computed(() => {
    const nodes = layoutNodes.value
    const edgeData = props.topology.edges || []
    
    return edgeData.map(edge => {
        const source = nodes.find(n => n.id === edge.source)
        const target = nodes.find(n => n.id === edge.target)
        if (!source || !target) return null
        return {
            x1: source.x + 80, // mid-right
            y1: source.y + 30, // mid-point
            x2: target.x - 10, // mid-left
            y2: target.y + 30
        }
    }).filter(e => e !== null)
})
</script>

<template>
  <div class="bg-[#111] border border-[#222] rounded-xl p-6 relative overflow-hidden">
    <div class="flex items-center justify-between mb-6">
        <h3 class="text-sm font-bold text-gray-400 uppercase tracking-widest flex items-center gap-2">
            <Activity class="w-4 h-4 text-emerald-500" /> Pipeline Topology Map
        </h3>
        <div class="flex gap-4 text-[10px] uppercase font-bold text-gray-600">
            <div class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-blue-500"></span> Ingest</div>
            <div class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-orange-500"></span> Train</div>
            <div class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-emerald-500"></span> Serve</div>
        </div>
    </div>

    <!-- SVG Map -->
    <div class="h-48 w-full bg-black/20 rounded-lg p-4 relative overflow-x-auto scrollbar-hide">
        <svg class="w-full h-full min-w-[800px]" viewBox="0 0 1000 150">
            <!-- Edges (Traces) -->
            <g v-for="(edge, i) in edges" :key="'e'+i">
                <path :d="`M ${edge.x1} ${edge.y1} L ${edge.x2} ${edge.y2}`" 
                      stroke="rgba(255,255,255,0.05)" stroke-width="2" fill="none" />
                <circle :cx="edge.x1" :cy="edge.y1" r="2" fill="rgba(255,255,255,0.2)" />
                <circle :cx="edge.x2" :cy="edge.y2" r="2" fill="rgba(255,255,255,0.2)" />
                
                <!-- Animated Data Pulse -->
                <circle r="2" fill="orange">
                    <animateMotion :path="`M ${edge.x1} ${edge.y1} L ${edge.x2} ${edge.y2}`" 
                                   dur="3s" repeatCount="indefinite" :begin="i * 0.5 + 's'" />
                </circle>
            </g>

            <!-- Nodes -->
            <foreignObject v-for="node in layoutNodes" :key="node.id"
                           :x="node.x" :y="node.y" width="160" height="60">
                <div :class="['flex items-center gap-3 p-2 rounded-lg border h-full transition-all duration-300', getCategoryColor(node.category)]">
                    <div class="p-1.5 bg-black/40 rounded flex items-center justify-center">
                        <component :is="getIcon(node.category)" class="w-4 h-4" />
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="text-[10px] font-black truncate leading-tight uppercase tracking-tight">{{ node.label }}</div>
                        <div class="flex items-center gap-1 mt-0.5">
                            <CheckCircle v-if="node.status === 'ok'" class="w-2 h-2 text-emerald-500" />
                            <AlertCircle v-else class="w-2 h-2 text-amber-500" />
                            <span class="text-[8px] uppercase font-bold opacity-60">{{ node.status || 'Active' }}</span>
                        </div>
                    </div>
                </div>
            </foreignObject>
        </svg>
    </div>
  </div>
</template>

<style scoped>
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
</style>
