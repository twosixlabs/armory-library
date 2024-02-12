import { RouterLink } from 'vue-router';

export default {
    components: {
        RouterLink,
    },
    template: `
        <div class="flex flex-row">
            <h1 class="font-medium text-xl mr-4 my-auto text-twosix-black uppercase">
                <router-link :to="{ name: 'index' }">
                    Evaluation Report
                </router-link>
            </h1>
        </div>
        <div>
            <img src="./assets/img/armory.png" alt="Armory" class="h-12" />
        </div>
    `
};
