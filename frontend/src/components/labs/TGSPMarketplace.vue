<script setup>
import { ref, onMounted, computed } from 'vue'
import {
    Package, Upload, CheckCircle, Clock, XCircle,
    Download, Shield, RefreshCw, Search, Filter,
    FileArchive, Rocket, Tag, User, Calendar
} from 'lucide-vue-next'

const packages = ref([])
const loading = ref(true)
const uploading = ref(false)
const searchQuery = ref('')
const statusFilter = ref('all')
const showUploadModal = ref(false)
const selectedPackage = ref(null)
const errorMessage = ref('')

const uploadFile = ref(null)

const filteredPackages = computed(() => {
    return packages.value.filter(pkg => {
        const matchesSearch = pkg.filename?.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                              pkg.producer_id?.toLowerCase().includes(searchQuery.value.toLowerCase())
        const matchesStatus = statusFilter.value === 'all' || pkg.status === statusFilter.value
        return matchesSearch && matchesStatus
    })
})

const fetchPackages = async () => {
    loading.value = true
    errorMessage.value = ''
    try {
        const res = await fetch('/api/v1/tgsp/packages')
        if (res.ok) {
            packages.value = await res.json()
        } else {
            throw new Error('Backend unavailable')
        }
    } catch (e) {
        console.error("Failed to fetch packages", e)
        packages.value = []
        errorMessage.value = 'Unable to load TGSP packages. Check backend connectivity and authentication.'
    }
    loading.value = false
}

const uploadPackage = async () => {
    if (!uploadFile.value) {
        alert("Please select a .tgsp file to upload")
        return
    }

    uploading.value = true
    const formData = new FormData()
    formData.append('file', uploadFile.value)

    try {
        const res = await fetch('/api/v1/tgsp/upload', {
            method: 'POST',
            body: formData
        })

        if (res.ok) {
            const data = await res.json()
            packages.value.unshift(data)
            showUploadModal.value = false
            uploadFile.value = null
            alert("Package uploaded successfully! Verification in progress.")
        } else {
            const err = await res.json()
            alert(`Upload failed: ${err.detail}`)
        }
    } catch (e) {
        console.error("Failed to upload package", e)
        alert("Upload failed. Check console for details.")
    }
    uploading.value = false
}

const deployToFleet = async (pkg) => {
    const fleetId = prompt("Enter Fleet ID to deploy to:")
    const channel = prompt("Channel (stable/canary):", "stable")

    if (!fleetId || !channel) return

    try {
        const res = await fetch('/api/v1/tgsp/releases', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                package_id: pkg.id,
                fleet_id: fleetId,
                channel: channel,
                is_active: true
            })
        })

        if (res.ok) {
            alert(`Package deployed to fleet ${fleetId} on ${channel} channel!`)
        } else {
            const err = await res.json()
            alert(`Deployment failed: ${err.detail}`)
        }
    } catch (e) {
        console.error("Failed to deploy", e)
    }
}

const getStatusColor = (status) => {
    const colors = {
        'verified': 'text-green-500 bg-green-500/10 border-green-500/30',
        'uploaded': 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30',
        'rejected': 'text-red-500 bg-red-500/10 border-red-500/30'
    }
    return colors[status] || 'text-gray-500 bg-gray-500/10 border-gray-500/30'
}

const getStatusIcon = (status) => {
    if (status === 'verified') return CheckCircle
    if (status === 'rejected') return XCircle
    return Clock
}

const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file && file.name.endsWith('.tgsp')) {
        uploadFile.value = file
    } else {
        alert("Please select a valid .tgsp file")
    }
}

onMounted(fetchPackages)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[#333] pb-6">
       <div>
         <h2 class="text-2xl font-bold flex items-center gap-3">
             <Package class="w-7 h-7 text-purple-500" />
             TGSP Marketplace
         </h2>
         <span class="text-xs text-gray-500">TensorGuard Security Package Registry & Distribution</span>
       </div>
       <div class="flex gap-3">
           <button @click="fetchPackages" :disabled="loading" class="btn btn-secondary">
               <RefreshCw class="w-4 h-4" :class="loading ? 'animate-spin' : ''" />
           </button>
           <button @click="showUploadModal = true" class="btn btn-primary">
               <Upload class="w-4 h-4 mr-2" /> Upload Package
           </button>
       </div>
    </div>

    <!-- Search & Filter -->
    <div class="flex items-center gap-4">
        <div class="flex-1 relative">
            <Search class="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
                v-model="searchQuery"
                type="text"
                placeholder="Search packages by name or producer..."
                class="w-full bg-[#111] border border-[#333] rounded-lg pl-10 pr-4 py-2 text-sm focus:border-primary outline-none"
            />
        </div>
        <div class="flex items-center gap-2">
            <Filter class="w-4 h-4 text-gray-500" />
            <select v-model="statusFilter" class="bg-[#111] border border-[#333] rounded-lg px-4 py-2 text-sm focus:border-primary outline-none cursor-pointer">
                <option value="all">All Status</option>
                <option value="verified">Verified</option>
                <option value="uploaded">Pending</option>
                <option value="rejected">Rejected</option>
            </select>
        </div>
    </div>

    <!-- Stats Cards -->
    <div class="grid grid-cols-4 gap-4">
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Package class="w-5 h-5 text-purple-500" />
                <span class="text-xs text-gray-500 uppercase">Total Packages</span>
            </div>
            <div class="text-2xl font-bold text-white">{{ packages.length }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <CheckCircle class="w-5 h-5 text-green-500" />
                <span class="text-xs text-gray-500 uppercase">Verified</span>
            </div>
            <div class="text-2xl font-bold text-green-500">{{ packages.filter(p => p.status === 'verified').length }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Clock class="w-5 h-5 text-yellow-500" />
                <span class="text-xs text-gray-500 uppercase">Pending Review</span>
            </div>
            <div class="text-2xl font-bold text-yellow-500">{{ packages.filter(p => p.status === 'uploaded').length }}</div>
        </div>
        <div class="bg-[#111] border border-[#333] rounded-lg p-4">
            <div class="flex items-center gap-3 mb-2">
                <Shield class="w-5 h-5 text-blue-500" />
                <span class="text-xs text-gray-500 uppercase">PQC Signed</span>
            </div>
            <div class="text-2xl font-bold text-blue-500">{{ packages.filter(p => p.status === 'verified').length }}</div>
        </div>
    </div>

    <!-- Package List -->
    <div v-if="loading" class="flex justify-center py-12">
        <div class="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
    </div>

    <div v-else-if="filteredPackages.length === 0" class="text-center py-12">
        <Package class="w-12 h-12 text-gray-700 mx-auto mb-4" />
        <p v-if="errorMessage" class="text-gray-500">{{ errorMessage }}</p>
        <p v-else class="text-gray-500">No packages found</p>
    </div>

    <div v-else class="space-y-4">
        <div v-for="pkg in filteredPackages" :key="pkg.id"
             class="bg-[#0d1117] border border-[#30363d] rounded-lg overflow-hidden hover:border-primary/50 transition-colors">
            <div class="p-6">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex items-center gap-4">
                        <div class="w-14 h-14 rounded-lg bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 flex items-center justify-center">
                            <FileArchive class="w-7 h-7 text-purple-400" />
                        </div>
                        <div>
                            <h3 class="font-bold text-lg text-white">{{ pkg.filename }}</h3>
                            <div class="flex items-center gap-3 mt-1">
                                <div class="flex items-center gap-1 text-xs text-gray-500">
                                    <User class="w-3 h-3" />
                                    {{ pkg.producer_id }}
                                </div>
                                <div class="flex items-center gap-1 text-xs text-gray-500">
                                    <Calendar class="w-3 h-3" />
                                    {{ pkg.created_at.split('T')[0] }}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center gap-3">
                        <span :class="['text-[10px] font-bold uppercase px-2 py-1 rounded border flex items-center gap-1', getStatusColor(pkg.status)]">
                            <component :is="getStatusIcon(pkg.status)" class="w-3 h-3" />
                            {{ pkg.status }}
                        </span>
                    </div>
                </div>

                <!-- Package Details -->
                <div class="grid grid-cols-3 gap-4 mb-4">
                    <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                        <div class="text-[10px] text-gray-500 uppercase mb-1">Policy</div>
                        <div class="text-sm text-white font-mono">{{ pkg.policy_id }}</div>
                        <div class="text-[10px] text-gray-500">v{{ pkg.policy_version }}</div>
                    </div>
                    <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                        <div class="text-[10px] text-gray-500 uppercase mb-1">Payloads</div>
                        <div class="flex flex-wrap gap-1">
                            <span v-for="payload in pkg.metadata_json?.payloads || []" :key="payload"
                                  class="text-[10px] bg-[#0d1117] px-1.5 py-0.5 rounded text-gray-400">
                                {{ payload }}
                            </span>
                        </div>
                    </div>
                    <div class="bg-[#161b22] p-3 rounded border border-[#30363d]">
                        <div class="text-[10px] text-gray-500 uppercase mb-1">Compatible Models</div>
                        <div class="flex flex-wrap gap-1">
                            <span v-for="model in pkg.metadata_json?.base_models || []" :key="model"
                                  class="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded border border-primary/30">
                                {{ model }}
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Manifest Hash -->
                <div class="flex items-center justify-between text-xs">
                    <div class="flex items-center gap-2 text-gray-500">
                        <Shield class="w-3 h-3" />
                        <span class="font-mono">{{ pkg.manifest_hash }}</span>
                    </div>
                    <div class="flex gap-2">
                        <button v-if="pkg.status === 'verified'" @click="deployToFleet(pkg)" class="btn btn-sm btn-primary">
                            <Rocket class="w-3 h-3 mr-1" /> Deploy
                        </button>
                        <button class="btn btn-sm btn-secondary">
                            <Download class="w-3 h-3 mr-1" /> Download
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Upload Modal -->
    <div v-if="showUploadModal" class="fixed inset-0 bg-black/90 flex items-center justify-center z-50 backdrop-blur-sm p-4">
        <div class="bg-[#0f0f0f] border border-primary/30 w-full max-w-lg rounded-xl shadow-2xl overflow-hidden">
            <div class="p-6 border-b border-[#222] flex items-center gap-4">
                <Upload class="w-6 h-6 text-purple-500" />
                <div>
                    <h3 class="text-xl font-bold text-white">Upload TGSP Package</h3>
                    <p class="text-[10px] text-gray-500 uppercase mt-1">Share your verified model package with the community</p>
                </div>
            </div>
            <div class="p-6 space-y-4">
                <!-- File Drop Zone -->
                <div class="border-2 border-dashed border-[#333] rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                    <input
                        type="file"
                        accept=".tgsp"
                        @change="handleFileSelect"
                        class="hidden"
                        id="tgsp-upload"
                    />
                    <label for="tgsp-upload" class="cursor-pointer">
                        <FileArchive class="w-12 h-12 text-gray-600 mx-auto mb-3" />
                        <div v-if="uploadFile" class="text-primary font-bold">{{ uploadFile.name }}</div>
                        <div v-else>
                            <p class="text-gray-400 mb-1">Drop your .tgsp file here or click to browse</p>
                            <p class="text-xs text-gray-600">Max file size: 500MB</p>
                        </div>
                    </label>
                </div>

                <!-- Info Box -->
                <div class="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 text-xs text-blue-400">
                    <strong>TGSP Format:</strong> Your package must include a valid manifest.json, manifest.sig (PQC signature),
                    policy.rego, and encrypted weights. The system will automatically verify integrity and policy compliance.
                </div>
            </div>
            <div class="p-6 bg-[#141414] flex justify-end gap-3 border-t border-[#222]">
                <button @click="showUploadModal = false; uploadFile = null" class="text-xs font-bold text-gray-500 uppercase px-4 py-2 hover:text-white transition-colors">Cancel</button>
                <button
                    @click="uploadPackage"
                    :disabled="!uploadFile || uploading"
                    class="btn btn-primary"
                >
                    <Upload class="w-4 h-4 mr-2" :class="uploading ? 'animate-bounce' : ''" />
                    {{ uploading ? 'Uploading...' : 'Upload Package' }}
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<style scoped>
.btn {
  @apply px-4 py-2 rounded font-medium transition-colors duration-200 flex items-center;
}
.btn-primary {
  @apply bg-orange-600 text-white hover:bg-orange-700;
}
.btn-secondary {
  @apply border border-[#30363d] text-gray-300 hover:text-white hover:bg-[#161b22];
}
.btn-sm {
  @apply px-3 py-1.5 text-xs;
}
</style>
