<template>
  <div class="releases-rollback p-6">
    <div class="flex justify-between items-center mb-6">
      <h2 class="text-xl font-bold">Releases & Rollback</h2>
      <div v-if="loading" class="text-sm text-gray-500">Loading...</div>
    </div>

    <!-- Active Stable Release -->
    <div class="bg-green-50 border border-green-200 rounded-lg p-6 mb-8 relative">
      <div class="absolute top-4 right-4">
        <span class="px-3 py-1 bg-green-200 text-green-800 rounded-full text-xs font-bold uppercase tracking-wider">Stable</span>
      </div>
      <h3 class="text-lg font-semibold text-green-900 mb-2">Active Production Adapter</h3>
      
      <div v-if="route.active_adapter_id" class="grid grid-cols-2 gap-4">
        <div>
          <div class="text-sm text-gray-500">Adapter ID</div>
          <div class="font-mono text-lg">{{ route.active_adapter_id }}</div>
        </div>
        <div>
          <div class="text-sm text-gray-500">Promoted At</div>
          <div>{{ formatDate(route.last_loop_at) }}</div> <!-- Approximation if exact date not in route status -->
        </div>
      </div>
      <div v-else class="text-gray-500 italic">
        No stable adapter active. System using base model.
      </div>

      <!-- Rollback Action -->
      <div class="mt-6 pt-4 border-t border-green-200 flex justify-between items-center">
        <div class="text-sm text-gray-600">
          <span v-if="route.fallback_adapter_id">
            Previous Stable: <span class="font-mono">{{ route.fallback_adapter_id }}</span>
          </span>
          <span v-else>No fallback available</span>
        </div>
        <button 
          @click="confirmRollback"
          :disabled="!route.fallback_adapter_id"
          class="bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 rounded-md transition-colors text-sm font-medium flex items-center gap-2"
        >
          <span class="material-icons text-sm">history</span>
          Rollback to Previous
        </button>
      </div>
    </div>

    <!-- Canary Release -->
    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-8 relative opacity-75">
      <div class="absolute top-4 right-4">
        <span class="px-3 py-1 bg-yellow-200 text-yellow-800 rounded-full text-xs font-bold uppercase tracking-wider">Canary</span>
      </div>
      <h3 class="text-lg font-semibold text-yellow-900 mb-2">Canary Channel</h3>
      
      <div v-if="route.canary_adapter_id" class="grid grid-cols-2 gap-4">
        <div>
          <div class="text-sm text-gray-500">Adapter ID</div>
          <div class="font-mono text-lg">{{ route.canary_adapter_id }}</div>
        </div>
        <div class="flex items-center gap-2">
           <button @click="promoteToStable(route.canary_adapter_id)" class="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm">
             Promote to Stable
           </button>
           <button @click="viewDiff(route.canary_adapter_id)" class="text-blue-600 hover:text-blue-800 text-sm underline">
             View Diff
           </button>
        </div>
      </div>
      <div v-else class="text-gray-500 italic">
        No active canary candidate.
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useRoute } from 'vue-router';
import { useContinuousStore } from '../../stores/continuous';

const props = defineProps(['routeKey']);
const store = useContinuousStore();
const loading = ref(false);

const route = computed(() => {
  return store.routes.find(r => r.route_key === props.routeKey) || {};
});

const formatDate = (ts) => {
  if (!ts) return 'N/A';
  return new Date(ts).toLocaleString();
};

const confirmRollback = async () => {
    if (!confirm("Are you sure you want to rollback to the previous stable adapter? This takes effect immediately.")) return;
    
    loading.value = true;
    try {
        await store.rollbackRoute(props.routeKey);
        // Refresh
        await store.fetchRoutes();
    } finally {
        loading.value = false;
    }
};

const promoteToStable = async (adapterId) => {
    loading.value = true;
    try {
        await store.promoteAdapter(props.routeKey, adapterId, 'stable');
        await store.fetchRoutes();
    } finally {
        loading.value = false;
    }
}

const viewDiff = (adapterId) => {
    // Navigate to diff view or emit event
    // simple alert or proper navigation
    console.log("View diff for", adapterId);
}
</script>
