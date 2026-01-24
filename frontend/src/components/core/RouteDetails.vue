<template>
  <div class="route-details p-6" v-if="route">
    <header class="mb-8 border-b pb-4 flex justify-between items-start">
      <div>
        <div class="flex items-center gap-3 mb-2">
            <h1 class="text-3xl font-bold text-gray-900">{{ route.route_key }}</h1>
            <span :class="{'bg-green-100 text-green-800': route.enabled, 'bg-gray-100 text-gray-800': !route.enabled}" 
                  class="px-2 py-1 rounded text-xs font-bold uppercase">
                {{ route.enabled ? 'Active' : 'Disabled' }}
            </span>
            <span v-if="privacyMode === 'n2he'" class="bg-indigo-100 text-indigo-800 px-2 py-1 rounded text-xs font-bold uppercase flex items-center gap-1">
                <span class="material-icons text-xs">lock</span> N2HE
            </span>
        </div>
        <p class="text-gray-600">{{ route.description || 'No description provided' }}</p>
        <div class="text-xs text-gray-400 font-mono mt-1">Base: {{ route.base_model_ref }}</div>
      </div>
      
      <div class="flex gap-2">
          <button @click="router.push('/')" class="text-gray-500 hover:text-gray-700 px-3 py-2">
              Back to Dashboard
          </button>
          <button @click="downloadExport(routeKey)" class="bg-gray-200 hover:bg-gray-300 text-gray-700 px-3 py-2 rounded shadow-sm flex items-center gap-2">
              <span class="material-icons text-sm">download</span>
              Export
          </button>
          <button @click="store.runOnce(routeKey)" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded shadow-sm flex items-center gap-2">
              <span class="material-icons text-sm">play_arrow</span>
              Run Loop Now
          </button>
      </div>
    </header>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <!-- Left Column: Releases & State -->
      <div class="lg:col-span-1 space-y-8">
        <ReleasesRollback :routeKey="routeKey" />
        
        <!-- Policy Summary Card -->
        <div class="bg-white p-6 rounded-lg border border-gray-200">
            <h3 class="font-bold text-gray-700 mb-4">Policy Settings</h3>
            <div v-if="policy" class="space-y-2 text-sm">
                <div class="flex justify-between">
                    <span class="text-gray-500">Novelty Threshold</span>
                    <span class="font-mono">{{ policy.novelty_threshold }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-500">Forgetting Budget</span>
                    <span class="font-mono">{{ policy.forgetting_budget }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-500">Update Cadence</span>
                    <span class="capitalize">{{ policy.update_cadence }}</span>
                </div>
            </div>
            <div v-else class="text-gray-400 text-sm">Loading policy...</div>
        </div>
      </div>

      <!-- Right Column: Timeline -->
      <div class="lg:col-span-2">
        <RouteTimeline :routeKey="routeKey" />
      </div>
    </div>
  </div>
  <div v-else class="p-12 text-center text-gray-500">
      Loading route details...
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useContinuousStore } from '../../stores/continuous';
import ReleasesRollback from './ReleasesRollback.vue';
import RouteTimeline from './RouteTimeline.vue';  // Assuming this exists from Phase 0

const router = useRouter();
const routeParam = useRoute();
const store = useContinuousStore();
const routeKey = routeParam.params.id;

const route = computed(() => store.routeDetails[routeKey]?.route);
const policy = computed(() => store.routeDetails[routeKey]?.policy);
const privacyMode = computed(() => store.routeDetails[routeKey]?.feed?.privacy_mode);

const downloadExport = async (key) => {
    try {
        const spec = await store.exportRoute(key);
        // Trigger download
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(spec, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href",     dataStr);
        downloadAnchorNode.setAttribute("download", key + "-export.json");
        document.body.appendChild(downloadAnchorNode); // required for firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    } catch (e) {
        console.error("Export failed", e);
        alert("Export failed: " + e.message);
    }
};

onMounted(async () => {
    await store.fetchRouteDetails(routeKey);
});
</script>
