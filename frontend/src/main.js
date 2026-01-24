import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import { createRouter, createWebHistory } from 'vue-router'

// Route Components
import ContinuousDashboard from './components/core/ContinuousDashboard.vue'
import RouteDetails from './components/core/RouteDetails.vue'
// Verify if RouteWizard is a page or modal. Let's make it a page for now or component.
// Using Dashboard as main entry.

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            name: 'dashboard',
            component: ContinuousDashboard
        },
        {
            path: '/routes/:id',
            name: 'route-details',
            component: RouteDetails
        },
        {
            path: '/:pathMatch(.*)*',
            redirect: '/'
        }
    ]
})

const pinia = createPinia()
const app = createApp(App)

app.use(pinia)
app.use(router)
app.mount('#app')
