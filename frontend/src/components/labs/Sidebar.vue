<script setup>
/**
 * Sidebar - Simplified Navigation
 *
 * Streamlined to 5 main sections based on engineering workflows:
 * - Command Center (Dashboard)
 * - Models (Lifecycle management)
 * - Operations (Fleet & Training)
 * - Security (Identity, Keys, Compliance)
 * - Settings
 */
import {
    LayoutDashboard, Bot, Server, Shield, Settings,
    ChevronRight
} from 'lucide-vue-next'

const props = defineProps(['activeTab'])
const emit = defineEmits(['update:activeTab'])

const navItems = [
    {
        id: 'dashboard',
        label: 'Command Center',
        icon: LayoutDashboard,
        description: 'System overview'
    },
    {
        id: 'models',
        label: 'Models',
        icon: Bot,
        description: 'VLA registry, training, evaluation'
    },
    {
        id: 'operations',
        label: 'Operations',
        icon: Server,
        description: 'Fleet, monitoring, deployments'
    },
    {
        id: 'security',
        label: 'Security',
        icon: Shield,
        description: 'Identity, keys, compliance'
    },
    {
        id: 'settings',
        label: 'Settings',
        icon: Settings,
        description: 'Configuration'
    }
]
</script>

<template>
  <aside class="w-64 bg-[#0d1117] border-r border-[#30363d] fixed h-full z-20 flex flex-col">
    <!-- Logo Area -->
    <div class="h-16 flex items-center px-5 border-b border-[#30363d] flex-shrink-0">
      <div class="w-9 h-9 bg-gradient-to-br from-primary to-orange-700 rounded-lg mr-3 flex items-center justify-center font-bold text-white text-sm shadow-lg">
        TG
      </div>
      <div>
        <span class="font-bold text-white text-sm">TensorGuard</span>
        <div class="text-[10px] text-gray-500">Flow v2.1</div>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 px-3 py-4">
      <div class="space-y-1">
        <button
          v-for="item in navItems"
          :key="item.id"
          @click="emit('update:activeTab', item.id)"
          :class="[
            'w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200 group',
            activeTab === item.id
              ? 'bg-[#1f2428] border-l-2 border-primary text-white'
              : 'text-gray-400 hover:text-white hover:bg-[#161b22]'
          ]"
        >
          <component
            :is="item.icon"
            :class="[
              'w-5 h-5 flex-shrink-0 transition-colors',
              activeTab === item.id ? 'text-primary' : 'text-gray-500 group-hover:text-gray-400'
            ]"
          />
          <div class="flex-1 text-left">
            <div class="text-sm font-medium">{{ item.label }}</div>
            <div v-if="activeTab !== item.id" class="text-[10px] text-gray-600 group-hover:text-gray-500">
              {{ item.description }}
            </div>
          </div>
          <ChevronRight
            v-if="activeTab === item.id"
            class="w-4 h-4 text-primary"
          />
        </button>
      </div>
    </nav>

    <!-- System Status -->
    <div class="px-4 py-3 border-t border-[#30363d] flex-shrink-0">
      <div class="flex items-center gap-2 mb-2">
        <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
        <span class="text-[10px] text-gray-500 uppercase">System Healthy</span>
      </div>
      <div class="text-[10px] text-gray-600">
        12 fleets • 847 devices online
      </div>
    </div>

    <!-- User Profile -->
    <div class="p-4 border-t border-[#30363d] flex-shrink-0">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-full bg-gradient-to-br from-green-600 to-green-800 flex items-center justify-center text-white text-xs font-bold shadow-md">
          DF
        </div>
        <div class="flex-1">
          <div class="text-sm font-medium text-white">Daniel Foo</div>
          <div class="text-[10px] text-gray-500">Organization Admin</div>
        </div>
      </div>
    </div>
  </aside>
</template>
