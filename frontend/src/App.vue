<template>
  <div class="h-screen bg-[#0a0a0a] text-gray-100 flex overflow-hidden">
    <!-- Sidebar (Core Only) -->
    <aside class="w-64 bg-[#111] border-r border-[#222] flex flex-col">
       <div class="p-6 border-b border-[#222]">
         <h1 class="font-bold text-lg flex items-center gap-2">
            <div class="w-6 h-6 bg-orange-600 rounded"></div>
            TensorGuardFlow
         </h1>
         <div class="text-xs text-gray-500 mt-1">Continuous Control Plane</div>
       </div>

       <nav class="flex-1 p-4 space-y-1">
          <router-link to="/" class="nav-item">
              <span class="material-icons text-sm">dashboard</span> Dashboard
          </router-link>
          <!-- Add other Core links if needed, usually Dashboard links to details -->
       </nav>

       <div class="p-4 border-t border-[#222] text-xs text-center text-gray-600">
          v2.3.0 Core
       </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 overflow-auto">
       <router-view></router-view>
    </main>
  </div>
</template>

<style scoped>
.nav-item {
    @apply flex items-center gap-3 px-3 py-2 rounded text-gray-400 hover:bg-[#222] hover:text-white transition-colors;
}
.router-link-active {
    @apply bg-[#222] text-white font-medium;
}
</style>
